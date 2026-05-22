"""评分排序 —— 顺势核心:趋势 + 量能 + 承接 占大头,异动重罚。

综合分 = 趋势分 + 量能分 + 承接分 + 龙头分 - 风险扣分,裁剪到 [0, 100]。
每个子项都附 0~1 的归一化原始值,便于在界面上解释"为什么这个分"。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import datasource as ds


@dataclass
class TrendMetrics:
    uptrend: bool
    ma_bull: bool        # 均线多头排列 ma5>ma10>ma20
    above_ma20: bool     # 收盘价在 20 日均线上方
    slope_pct: float     # 近 20 日年化斜率(%/日 近似)
    raw: float = 0.0     # 0~1 趋势强度


def compute_trend(daily: pd.DataFrame) -> TrendMetrics:
    if daily is None or len(daily) < 20:
        return TrendMetrics(False, False, False, 0.0, 0.0)
    close = daily["close"].astype(float).reset_index(drop=True)
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    last = close.iloc[-1]

    ma_bull = ma5 > ma10 > ma20
    above_ma20 = last > ma20

    # 斜率:近 20 日线性回归的归一化日涨幅
    y = close.iloc[-20:].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    slope_pct = slope / max(y.mean(), 1e-6) * 100

    uptrend = ma_bull and above_ma20 and slope_pct > 0
    # 趋势强度归一:斜率贡献为主,均线结构加成
    raw = np.clip(slope_pct / 1.5, 0, 1) * 0.6
    raw += 0.25 if ma_bull else 0.0
    raw += 0.15 if above_ma20 else 0.0
    return TrendMetrics(uptrend, ma_bull, above_ma20, round(slope_pct, 3), round(min(raw, 1.0), 3))


def _volume_raw(row, trend_up: bool) -> float:
    """量能分:温和放量(量比 1.0~2.5)最佳;过度放量(>3.5)可能赶顶降分;
    缩量在上升趋势中(回踩)给中性偏上。"""
    vr = float(row.get(ds.COL_VOL_RATIO, 1) or 1)
    if vr >= 3.5:
        return 0.45            # 过度放量,警惕
    if 1.2 <= vr <= 2.5:
        return 1.0             # 理想温和放量
    if 1.0 <= vr < 1.2:
        return 0.8
    # vr < 1.0 缩量
    return 0.65 if trend_up else 0.4


def _support_raw(row) -> float:
    """承接力度代理:用换手率(适中=有承接但不过热)与振幅(越小越稳)。
    真实数据可进一步接入委买委卖/内外盘;此处用现货快照可得字段近似。"""
    to = float(row.get(ds.COL_TURNOVER, 0) or 0)
    amp = float(row.get(ds.COL_AMPLITUDE, 0) or 0)
    # 换手 3~12% 视为活跃且有承接
    if 3 <= to <= 12:
        to_score = 1.0
    elif to < 3:
        to_score = 0.6
    else:  # 过热
        to_score = 0.5
    # 振幅惩罚:振幅越大承接越不稳
    amp_score = float(np.clip(1 - (amp - 4) / 12, 0.3, 1.0))
    return round(0.6 * to_score + 0.4 * amp_score, 3)


def _risk_penalty(row, cfg: dict) -> tuple[float, list[str]]:
    rp = cfg["risk_penalty"]
    pct = float(row[ds.COL_PCT])
    amp = float(row.get(ds.COL_AMPLITUDE, 0) or 0)
    penalty = 0.0
    flags = []
    # 涨停判定:主板~10%,创业板/科创板~20%。用代码前缀粗略区分。
    code = str(row[ds.COL_CODE])
    limit = 20 if code[:2] in ("30", "68") else 10
    if pct >= limit - 0.3:
        penalty += rp["limit_up_today"]; flags.append("涨停-警惕赶顶")
    elif pct >= 9:
        penalty += rp["near_limit_up"]; flags.append("接近涨停")
    if pct <= -5:
        penalty += rp["big_drop_today"]; flags.append("大跌-趋势存疑")
    if amp > 12:
        penalty += rp["high_volatility"]; flags.append("振幅过大")
    return penalty, flags


@dataclass
class ScoreResult:
    code: str
    name: str
    score: float
    trend_score: float
    volume_score: float
    support_score: float
    leader_score: float
    penalty: float
    uptrend: bool
    leader_label: str
    sector: str
    flags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def score_universe(df: pd.DataFrame, source: ds.DataSource, cfg: dict) -> pd.DataFrame:
    """对已通过筛选并标注龙头的 df 逐只打分,返回按综合分降序的 DataFrame。"""
    w = cfg["score_weights"]
    hist_days = cfg["runtime"]["hist_days"]
    results = []

    for _, row in df.iterrows():
        code = str(row[ds.COL_CODE])
        daily = source.daily(code, hist_days)
        tm = compute_trend(daily)

        vol_raw = _volume_raw(row, tm.uptrend)
        sup_raw = _support_raw(row)

        lr = int(row.get("leader_rank", -1))
        leader_base = {0: 1.0, 1: 0.7, 2: 0.45}.get(lr, 0.0)
        # 连板天梯加成:属于今日活跃板块的龙头才是真龙头。
        # 冷门板块龙头打 7 折,热门板块龙头按热度上浮(最高 1.0)。
        hot_score = float(row.get("hot_score", 0) or 0)
        leader_raw = leader_base * (0.7 + 0.3 * min(1.0, hot_score * 1.5))

        trend_score = tm.raw * w["trend"]
        volume_score = vol_raw * w["volume"]
        support_score = sup_raw * w["bid_support"]
        leader_score = leader_raw * w["leader"]

        penalty, flags = _risk_penalty(row, cfg)

        total = trend_score + volume_score + support_score + leader_score + penalty
        total = float(np.clip(total, 0, 100))

        results.append(ScoreResult(
            code=code,
            name=str(row[ds.COL_NAME]),
            score=round(total, 1),
            trend_score=round(trend_score, 1),
            volume_score=round(volume_score, 1),
            support_score=round(support_score, 1),
            leader_score=round(leader_score, 1),
            penalty=round(penalty, 1),
            uptrend=tm.uptrend,
            leader_label=str(row.get("leader_label", "")),
            sector=str(row.get("sector", "未分类")),
            flags=flags,
            detail={
                "price": float(row[ds.COL_PRICE]),
                "pct_chg": float(row[ds.COL_PCT]),
                "vol_ratio": float(row.get(ds.COL_VOL_RATIO, 0) or 0),
                "turnover": float(row.get(ds.COL_TURNOVER, 0) or 0),
                "mkt_cap": float(row.get(ds.COL_MKT_CAP, 0) or 0),
                "ma_bull": tm.ma_bull,
                "above_ma20": tm.above_ma20,
                "slope_pct": tm.slope_pct,
                # 4 个归一化原始值(与权重无关),供进化引擎用任意权重瞬间重算分数
                "raw_trend": round(float(tm.raw), 4),
                "raw_volume": round(float(vol_raw), 4),
                "raw_support": round(float(sup_raw), 4),
                "raw_leader": round(float(leader_raw), 4),
                "raw_penalty": round(float(penalty), 4),
                # 连板天梯热度(界面展示用)
                "sector_hot": bool(row.get("sector_hot", False)),
                "hot_score": round(float(row.get("hot_score", 0) or 0), 3),
            },
        ))

    out = pd.DataFrame([r.__dict__ for r in results])
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).reset_index(drop=True)
