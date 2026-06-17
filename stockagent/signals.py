"""买入信号判定(只提示,不下单)。

买入纪律:顺势不逆 / 不反包 / 不突破 / 警惕涨停 / 确认量能承接。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BuyVerdict:
    code: str
    name: str
    can_buy: bool
    reasons: list[str]


def evaluate_buy(scored_row: pd.Series, cfg: dict) -> BuyVerdict:
    bs = cfg["buy_signal"]
    reasons = []
    ok = True

    if scored_row["score"] < bs["min_score"]:
        ok = False
        reasons.append(f"评分 {scored_row['score']} < 阈值 {bs['min_score']}")

    if bs.get("require_uptrend", True) and not scored_row["uptrend"]:
        ok = False
        reasons.append("非明确上升趋势(不买)")

    vr = float(scored_row["detail"].get("vol_ratio", 0) or 0)
    if vr < bs.get("min_volume_ratio", 1.0):
        ok = False
        reasons.append(f"量比 {vr} 不足,承接存疑")

    flags = scored_row.get("flags", []) or []
    if bs.get("reject_breakout", True) and "涨停-警惕赶顶" in flags:
        ok = False
        reasons.append("涨停异动,警惕赶顶,不追")

    if ok:
        reasons.append("顺势 + 有量承接,符合买入纪律")
    return BuyVerdict(scored_row["code"], scored_row["name"], ok, reasons)
