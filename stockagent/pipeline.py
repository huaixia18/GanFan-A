"""全流程串联:行情 → 筛选 → 龙头 → 评分排序 → 买入判定,并叠加风控状态。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import datasource as ds
from . import leaders, screener, signals
from .risk import RiskState
from .scorer import score_universe

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
STATE_PATH = ROOT / "state.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class PipelineResult:
    mode: str                     # tencent / eastmoney
    risk_status: str              # ACTIVE / RESTING
    resting_days_left: int
    last_drawdown: float | None
    ranked: list[dict] = field(default_factory=list)   # 排序后的选股
    dropped_count: int = 0
    universe_count: int = 0
    weights_version: str = "config"


def run(cfg: dict | None = None, state: RiskState | None = None) -> PipelineResult:
    cfg = cfg or load_config()
    state = state or RiskState.load(STATE_PATH)

    # 自选池:config 与 state 合并(state 运行时可增删)
    wl = list({*(str(c) for c in cfg.get("watchlist", []) or []), *state.watchlist})
    cfg = {**cfg, "watchlist": wl}

    rt = cfg["runtime"]
    full_market = rt.get("full_market", False)

    if full_market:
        source = ds.DataSource(
            full_market=True,
            prefer=rt.get("prefer_source", "tencent"),
        )
        sector_map_file = {}
    else:
        codes, sector_map_file = ds.load_universe_with_sectors(wl)
        source = ds.DataSource(
            universe=codes,
            prefer=rt.get("prefer_source", "tencent"),
        )

    spot = source.spot()
    universe_count = len(spot)

    passed, dropped = screener.apply_filters(spot, cfg)

    # 全市场模式:基本面筛后按活跃度(涨幅+量比)粗筛到 prefilter_top,只对其算趋势(控速)
    if full_market and not passed.empty:
        prefilter_top = rt.get("prefilter_top", 200)
        activity = passed[ds.COL_PCT] + 2.0 * passed[ds.COL_VOL_RATIO].clip(lower=0)
        passed = passed.assign(_act=activity).sort_values(
            "_act", ascending=False).head(prefilter_top).drop(columns="_act").reset_index(drop=True)

    # 板块归属:全市场/腾讯用行业映射;东财源尝试在线板块接口
    sector_map = None
    if full_market and source.is_real:
        sector_map = source.industry_map
    elif source.mode == "tencent":
        sector_map = sector_map_file
    elif source.mode == "eastmoney":
        sector_map = leaders.build_sector_map(passed[ds.COL_CODE].astype(str).tolist())
    annotated = leaders.annotate_leaders(passed, sector_map)

    # 进化:用当前激活的权重版本覆盖 config 死值(无历史则用 config)
    from . import evolve
    active_w = evolve.active_weights(cfg["score_weights"])
    weights_version = evolve.status()["active"]
    weights_version = weights_version["id"] if weights_version else "config"
    cfg = {**cfg, "score_weights": {**cfg["score_weights"], **active_w}}

    ranked_df = score_universe(annotated, source, cfg)
    top_n = rt["top_n"]
    ranked_df = ranked_df.head(top_n) if not ranked_df.empty else ranked_df

    can_trade = state.can_trade()
    ranked = []
    if not ranked_df.empty:
        for _, r in ranked_df.iterrows():
            verdict = signals.evaluate_buy(r, cfg)
            ranked.append({
                **{k: r[k] for k in ["code", "name", "score", "trend_score",
                                     "volume_score", "support_score", "leader_score",
                                     "penalty", "uptrend", "leader_label", "sector",
                                     "flags", "detail"]},
                "can_buy": verdict.can_buy and can_trade,
                "buy_reasons": verdict.reasons,
                # 休息期强制屏蔽买入
                "buy_blocked_by_rest": verdict.can_buy and not can_trade,
            })

    # 记录本次高分股快照,供后续 N 日收益验证与进化(仅真实源)
    if ranked and source.is_real:
        try:
            evolve.record_predictions(ranked, weights_version)
        except Exception:  # noqa: BLE001
            pass

    return PipelineResult(
        mode=source.mode,
        risk_status=state.status(),
        resting_days_left=state.resting_days_left(),
        last_drawdown=state.last_drawdown,
        ranked=ranked,
        dropped_count=len(dropped),
        universe_count=universe_count,
        weights_version=weights_version,
    )
