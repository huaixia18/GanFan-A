"""自主进化引擎 —— 在严格安全闸内自动优化【评分权重】。

设计哲学(为什么这么克制):
  全自动调参 + 单一收益目标是量化最大的亏钱陷阱(过拟合 / 目标错配 / 小样本)。
  因此这里只让系统优化"安全的部分",纪律红线永不参与:

  可优化:score_weights 的 4 个权重(trend / volume / bid_support / leader)。
  锁死(永不动):市值/股价/排除亏损/不追涨停(risk_penalty)/买入阈值/-4.5% 风控。

三道安全闸:
  1) 样本门槛:已验证样本 < MIN_SAMPLES 时拒绝调参,只记录。
  2) 小步搜索:每个权重只在当前值 ±STEP_RATIO 内微调,且保持总和=100。
  3) 留痕 + 回滚:每版权重存档;新版若在验证集上不如历史最佳,自动回滚到最佳版本。

评判标准:IC(分数与未来 N 日收益的秩相关)为主,辅以高分组 vs 低分组的收益差。
  —— 衡量的是"高分是否真的预示更高收益",不是"绝对收益最大化"(避免诱导违纪)。

数据文件(均不入库,运行时生成):
  predictions.json  每次选股的高分股快照 + 待验证/已回填的 N 日收益
  evolution.json    权重版本历史 + 每版评估分 + 当前激活版本
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from . import datasource as ds
from .scorer import compute_trend  # noqa: F401  (保留,便于未来扩展)

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "predictions.json"
EVOLUTION_PATH = ROOT / "evolution.json"

# ---- 安全闸参数 ----
MIN_SAMPLES = 30          # 已验证样本不足此数,拒绝调参
STEP_RATIO = 0.15         # 每次权重微调幅度上限(±15%)
HORIZON_DAYS = 5          # 评判用的"N 日收益"主口径
RECORD_TOP = 10           # 每次选股记录前 N 只用于后续验证
WEIGHT_KEYS = ["trend", "volume", "bid_support", "leader"]
RAW_KEYS = {"trend": "raw_trend", "volume": "raw_volume",
            "bid_support": "raw_support", "leader": "raw_leader"}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _load(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return default
    return default


def _save(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _score_from_raw(detail: dict, weights: dict) -> float:
    """用任意权重 + 已存的归一化 raw 值重算综合分(无需重新拉行情)。"""
    s = 0.0
    for k in WEIGHT_KEYS:
        s += float(detail.get(RAW_KEYS[k], 0)) * float(weights[k])
    s += float(detail.get("raw_penalty", 0))   # 风险扣分(锁死,照搬)
    return max(0.0, min(100.0, s))


# --------------------------------------------------------------------------- #
# 1) 记录:每次选股存档高分股快照
# --------------------------------------------------------------------------- #
def record_predictions(ranked: list[dict], weights_version: str,
                       today: date | None = None):
    """ranked 为 pipeline 输出的选股列表(含 detail 里的 raw_* 值)。"""
    today = (today or date.today()).isoformat()
    data = _load(PREDICTIONS_PATH, {"records": []})
    existing = {(r["date"], r["code"]) for r in data["records"]}
    for item in ranked[:RECORD_TOP]:
        key = (today, str(item["code"]))
        if key in existing:
            continue
        d = item.get("detail", {})
        data["records"].append({
            "date": today,
            "code": str(item["code"]),
            "name": item.get("name", ""),
            "score": item.get("score"),
            "entry_price": d.get("price"),
            "raw": {k: d.get(RAW_KEYS[k]) for k in WEIGHT_KEYS} | {"penalty": d.get("raw_penalty")},
            "weights_version": weights_version,
            "verified": False,
            "ret": {},          # 回填 {"1": x%, "3": y%, "5": z%}
        })
    _save(PREDICTIONS_PATH, data)
    return len(data["records"])


# --------------------------------------------------------------------------- #
# 2) 收益回收:对到期预测拉真实 N 日收益
# --------------------------------------------------------------------------- #
def verify_returns(source: ds.DataSource, horizons=(1, 3, 5),
                   today: date | None = None) -> int:
    today = today or date.today()
    data = _load(PREDICTIONS_PATH, {"records": []})
    newly = 0
    for rec in data["records"]:
        if rec.get("verified"):
            continue
        d0 = date.fromisoformat(rec["date"])
        if (today - d0).days < max(horizons) + 1:
            continue   # 还没到最长 horizon,等
        try:
            daily = source.daily(rec["code"], 120)
        except Exception:  # noqa: BLE001
            continue
        if daily is None or daily.empty:
            continue
        daily = daily.reset_index(drop=True)
        # 找到 entry 当天(或之后第一个交易日)的索引
        dates = list(daily["date"].astype(str))
        idx = next((i for i, dt in enumerate(dates) if dt >= rec["date"]), None)
        if idx is None:
            continue
        base = float(daily["close"].iloc[idx])
        ret = {}
        for h in horizons:
            j = idx + h
            if j < len(daily) and base > 0:
                ret[str(h)] = round((float(daily["close"].iloc[j]) - base) / base * 100, 2)
        if ret:
            rec["ret"] = ret
            rec["verified"] = True
            newly += 1
    _save(PREDICTIONS_PATH, data)
    return newly


# --------------------------------------------------------------------------- #
# 3) 评估:某套权重在已验证样本上的好坏
# --------------------------------------------------------------------------- #
def _spearman_ic(xs: list[float], ys: list[float]) -> float:
    """分数与收益的秩相关(Spearman),衡量"高分是否预示高收益"。"""
    n = len(xs)
    if n < 5:
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return round(num / (dx * dy), 4) if dx and dy else 0.0


def evaluate(weights: dict, horizon: int = HORIZON_DAYS) -> dict:
    """用已验证样本评估一套权重:返回 IC、分组收益差、样本数。"""
    data = _load(PREDICTIONS_PATH, {"records": []})
    samples = [r for r in data["records"]
               if r.get("verified") and str(horizon) in r.get("ret", {})]
    n = len(samples)
    if n < 5:
        return {"n": n, "ic": 0.0, "spread": 0.0, "avg_ret": 0.0}

    scores, rets = [], []
    for r in samples:
        d = {RAW_KEYS[k]: r["raw"].get(k, 0) for k in WEIGHT_KEYS}
        d["raw_penalty"] = r["raw"].get("penalty", 0)
        scores.append(_score_from_raw(d, weights))
        rets.append(r["ret"][str(horizon)])

    ic = _spearman_ic(scores, rets)
    # 高分组(前1/3) vs 低分组(后1/3)平均收益差
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = max(1, n // 3)
    hi = sum(rets[i] for i in order[:k]) / k
    lo = sum(rets[i] for i in order[-k:]) / k
    return {
        "n": n,
        "ic": ic,
        "spread": round(hi - lo, 2),     # 高分组比低分组多赚几个点
        "avg_ret": round(sum(rets) / n, 2),
    }


# --------------------------------------------------------------------------- #
# 4) 进化引擎:安全闸内搜索更优权重 + 自动回滚
# --------------------------------------------------------------------------- #
def _normalize_to_100(w: dict) -> dict:
    total = sum(w[k] for k in WEIGHT_KEYS)
    if total <= 0:
        return {k: 25.0 for k in WEIGHT_KEYS}
    return {k: round(w[k] / total * 100, 2) for k in WEIGHT_KEYS}


def _candidates(base: dict) -> list[dict]:
    """在 ±STEP_RATIO 内对每个权重 ±一步,生成小步邻域候选(保持总和=100)。"""
    cands = []
    for k in WEIGHT_KEYS:
        for sign in (+1, -1):
            w = dict(base)
            w[k] = max(1.0, base[k] * (1 + sign * STEP_RATIO))
            cands.append(_normalize_to_100(w))
    return cands


def _quality(metrics: dict) -> float:
    """单一质量分:以 IC 为主(预测力),收益差为辅。"""
    return metrics["ic"] * 1.0 + metrics["spread"] * 0.02


def evolve(current_weights: dict, today: date | None = None) -> dict:
    """主进化动作。返回报告 dict(无论是否调整都会写 evolution.json)。"""
    today = (today or date.today()).isoformat()
    evo = _load(EVOLUTION_PATH, {"versions": [], "active": None})

    base = _normalize_to_100({k: float(current_weights[k]) for k in WEIGHT_KEYS})
    base_metrics = evaluate(base)
    n = base_metrics["n"]

    report = {
        "date": today, "samples": n, "min_samples": MIN_SAMPLES,
        "current_weights": base, "current_metrics": base_metrics,
        "action": None, "new_weights": None, "new_metrics": None, "note": "",
    }

    # 安全闸 1:样本不足,只记录不动手
    if n < MIN_SAMPLES:
        report["action"] = "hold"
        report["note"] = f"已验证样本 {n} < {MIN_SAMPLES},暂不调参(避免小样本过拟合)。"
        _record_version(evo, base, base_metrics, today, activate=evo["active"] is None)
        _save(EVOLUTION_PATH, evo)
        return report

    # 在小步邻域里找质量更高的权重
    best_w, best_m, best_q = base, base_metrics, _quality(base_metrics)
    for cand in _candidates(base):
        m = evaluate(cand)
        if _quality(m) > best_q:
            best_w, best_m, best_q = cand, m, _quality(m)

    if best_w == base:
        report["action"] = "hold"
        report["note"] = "邻域内未找到更优权重,保持当前。"
        _record_version(evo, base, base_metrics, today, activate=False)
        _save(EVOLUTION_PATH, evo)
        return report

    # 安全闸 3:与历史最佳比较,新版必须不差于历史最佳才采纳;否则回滚
    hist_best = _historical_best(evo)
    if hist_best and _quality(hist_best["metrics"]) > best_q:
        # 自动回滚:激活历史最佳版本
        report["action"] = "rollback"
        report["new_weights"] = hist_best["weights"]
        report["new_metrics"] = hist_best["metrics"]
        report["note"] = "新候选不如历史最佳,自动回滚到最佳版本。"
        evo["active"] = hist_best["id"]
        _save(EVOLUTION_PATH, evo)
        return report

    # 采纳新权重
    report["action"] = "evolve"
    report["new_weights"] = best_w
    report["new_metrics"] = best_m
    report["note"] = (f"IC {base_metrics['ic']}→{best_m['ic']}, "
                      f"收益差 {base_metrics['spread']}→{best_m['spread']}。已小步进化。")
    ver = _record_version(evo, best_w, best_m, today, activate=True)
    report["new_version"] = ver
    _save(EVOLUTION_PATH, evo)
    return report


def _record_version(evo: dict, weights: dict, metrics: dict, day: str,
                    activate: bool) -> str:
    vid = f"w{len(evo['versions'])+1:03d}"
    evo["versions"].append({
        "id": vid, "date": day, "weights": weights, "metrics": metrics,
    })
    if activate:
        evo["active"] = vid
    return vid


def _historical_best(evo: dict) -> dict | None:
    cands = [v for v in evo["versions"] if v["metrics"].get("n", 0) >= MIN_SAMPLES]
    if not cands:
        return None
    return max(cands, key=lambda v: _quality(v["metrics"]))


def active_weights(fallback: dict) -> dict:
    """返回当前激活版本的权重;无则用 config 里的 fallback。"""
    evo = _load(EVOLUTION_PATH, {"versions": [], "active": None})
    if evo.get("active"):
        for v in evo["versions"]:
            if v["id"] == evo["active"]:
                return v["weights"]
    return {k: float(fallback[k]) for k in WEIGHT_KEYS}


def status() -> dict:
    """界面用:当前版本 / 样本数 / IC / 历史记录。"""
    evo = _load(EVOLUTION_PATH, {"versions": [], "active": None})
    preds = _load(PREDICTIONS_PATH, {"records": []})
    verified = [r for r in preds["records"] if r.get("verified")]
    active = next((v for v in evo["versions"] if v["id"] == evo.get("active")), None)
    return {
        "active": active,
        "versions": evo["versions"][-10:],
        "total_predictions": len(preds["records"]),
        "verified_samples": len(verified),
        "min_samples": MIN_SAMPLES,
    }
