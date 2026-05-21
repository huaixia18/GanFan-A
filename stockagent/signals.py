"""买入信号 + 分时卖出/止损规则提示(只提示,不下单)。

买入纪律:顺势不逆 / 不反包 / 不突破 / 警惕涨停 / 确认量能承接。
卖出纪律:急拉后 1 分钟拉不回 → 卖;缓拉后急跌 2 次回不来 → 卖。
分时卖出依赖盘中 tick,这里给出可执行的判定函数(供盘中喂数据),
以及对当前候选给出"是否够买入条件"的静态判断。
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


# --------------------------------------------------------------------------- #
# 分时卖出/止损 —— 喂入当日逐分钟价格序列(list[float]),返回是否触发卖出
# --------------------------------------------------------------------------- #
def intraday_sell_signal(minute_prices: list[float]) -> tuple[bool, str]:
    """规则一:急拉后快速下跌,1 分钟内无法拉回 → 卖。
       规则二:缓拉后出现急跌,2 次尝试无法有效回升 → 卖。
    minute_prices: 当日分时收盘价序列(至少需要数个点)。
    """
    if len(minute_prices) < 4:
        return False, "数据不足"
    p = minute_prices

    # 规则一:识别急拉(单分钟涨幅 > 2%)后下一分钟急跌且未拉回
    for i in range(1, len(p) - 1):
        spike = (p[i] - p[i - 1]) / p[i - 1]
        if spike > 0.02:  # 急拉
            drop = (p[i + 1] - p[i]) / p[i]
            if drop < -0.015:  # 急跌
                # 1 分钟内是否拉回(看再下一分钟)
                if i + 2 >= len(p) or p[i + 2] < p[i]:
                    return True, "急拉后快速下跌,1分钟内未拉回 → 卖出(疑诱多出货)"

    # 规则二:从高点缓拉后急跌,统计两次反弹失败
    peak = max(p)
    peak_i = p.index(peak)
    after = p[peak_i:]
    if len(after) >= 4:
        drop_from_peak = (after[-1] - peak) / peak
        if drop_from_peak < -0.02:
            # 数反弹尝试次数:局部上行后又被压回
            attempts = 0
            for j in range(1, len(after) - 1):
                if after[j] > after[j - 1] and after[j + 1] < after[j]:
                    attempts += 1
            if attempts >= 2:
                return True, "缓拉后急跌,2次反弹失败 → 卖出(动能衰竭)"

    return False, "未触发卖出信号"
