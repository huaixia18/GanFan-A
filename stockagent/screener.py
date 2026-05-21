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


def apply_filters(spot: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (通过的, 被剔除的带原因)。"""
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
