"""基本面硬性筛选 —— 高度克制,先把不做的票全部剔除。

对应纪律:
  - 排除亏损票(PE<0 视为亏损代理)
  - 市值 <= 100 亿
  - 股价 5~30 元
  - 排除 ST / *ST
  - 排除次新(可选)
  - 自选池限定:只保留池内股票(池为空则不限制)
每条剔除都记录原因,便于复盘"为什么没选它"。
"""
from __future__ import annotations

import pandas as pd

from . import datasource as ds

# 板块识别(按代码前缀)
BOARD_LABELS = {"chinext": "创业板", "star": "科创板", "bse": "北交所", "main": "主板"}


def board_of(code: str) -> str:
    """按代码前缀判断板块:chinext/star/bse/main。"""
    code = str(code)
    if code.startswith("30"):
        return "chinext"        # 创业板,20% 涨跌停
    if code.startswith("688"):
        return "star"           # 科创板,20% 涨跌停
    if code.startswith(("4", "8", "92")):
        return "bse"            # 北交所,30% 涨跌停(83/87/88/920 等)
    return "main"               # 主板(沪 60 / 深 000,001,002,003)


def apply_filters(spot: pd.DataFrame, cfg: dict,
                  sector_map: dict | None = None,
                  active_sectors: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (通过的, 被剔除的带原因)。

    active_sectors 非空时启用「活跃板块硬筛」:只保留所属细分行业在该集合里的股票。
    需要 sector_map(代码→细分行业)。两者任一缺失则跳过此筛(降级,不硬筛)。
    """
    f = cfg["fundamental"]
    watchlist = [str(c) for c in cfg.get("watchlist", []) or []]

    df = spot.copy()
    df["_drop_reason"] = ""

    def mark(mask: pd.Series, reason: str):
        # 只给尚未被标记的行打原因,保留首个命中的剔除理由
        newly = mask & (df["_drop_reason"] == "")
        df.loc[newly, "_drop_reason"] = reason

    if f.get("exclude_st", True):
        st_mask = df[ds.COL_NAME].astype(str).str.contains("ST", case=False, na=False)
        mark(st_mask, "ST/*ST")

    # 板块过滤:排除高波动板块(创业板/科创板/北交所)
    exclude_boards = f.get("exclude_boards", []) or []
    if exclude_boards:
        board_series = df[ds.COL_CODE].astype(str).map(board_of)
        for b in exclude_boards:
            mark(board_series == b, f"排除{BOARD_LABELS.get(b, b)}")

    # 活跃板块硬筛:只保留所属细分行业在今日热度 Top N 里的股票(跟踪活跃板块的纪律)
    # 需 sector_map + active_sectors;缺任一则不硬筛(降级)。
    if active_sectors and sector_map:
        sec = df[ds.COL_CODE].astype(str).map(lambda c: sector_map.get(c.zfill(6)))
        mark(~sec.isin(active_sectors), "非活跃板块")

    if f.get("exclude_loss", True) and ds.COL_PE in df.columns:
        mark(df[ds.COL_PE] < 0, "亏损(PE<0)")

    if ds.COL_MKT_CAP in df.columns:
        mark(df[ds.COL_MKT_CAP] > f["market_cap_max"], f"市值>{f['market_cap_max']}亿")

    mark(df[ds.COL_PRICE] < f["price_min"], f"股价<{f['price_min']}")
    mark(df[ds.COL_PRICE] > f["price_max"], f"股价>{f['price_max']}")

    if watchlist:
        mark(~df[ds.COL_CODE].astype(str).isin(watchlist), "不在自选池")

    passed = df[df["_drop_reason"] == ""].drop(columns=["_drop_reason"]).reset_index(drop=True)
    dropped = df[df["_drop_reason"] != ""][[ds.COL_CODE, ds.COL_NAME, "_drop_reason"]].reset_index(drop=True)
    return passed, dropped
