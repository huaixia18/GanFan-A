"""板块龙头识别 —— 龙头是市场选出来的,不是猜的。

做法:把股票按板块分组,组内按"当日涨幅 + 量能"综合排序,
取前三标记为龙一/龙二/龙三。优先关注龙头。

板块归属来源:外部传入的 code->sector 映射
  - 全市场/腾讯模式:用交易所行业 / universe.txt 分组;
  - 东财模式:build_sector_map 在线构造;
  取不到时整体归为「未分类」,龙头标记降级为按全市场涨幅排名。
"""
from __future__ import annotations

import pandas as pd

from . import datasource as ds

LEADER_LABELS = {0: "龙一", 1: "龙二", 2: "龙三"}


def _activity_score(row) -> float:
    # 龙头判定:涨幅为主,量比为辅(有量的领涨才是真龙头)
    return float(row[ds.COL_PCT]) + 2.0 * float(row.get(ds.COL_VOL_RATIO, 1) or 1)


def annotate_leaders(df: pd.DataFrame, sector_map: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    if "_sector" not in out.columns:
        if sector_map:
            out["_sector"] = out[ds.COL_CODE].astype(str).map(sector_map).fillna("未分类")
        else:
            out["_sector"] = "未分类"

    out["_activity"] = out.apply(_activity_score, axis=1)
    out["leader_rank"] = -1   # -1 表示非前三
    out["leader_label"] = ""
    out["sector"] = out["_sector"]

    for _, idx in out.groupby("_sector").groups.items():
        sub = out.loc[idx].sort_values("_activity", ascending=False)
        for r, (i, _) in enumerate(sub.iterrows()):
            if r < 3:
                out.at[i, "leader_rank"] = r
                out.at[i, "leader_label"] = LEADER_LABELS[r]

    return out.drop(columns=["_sector", "_activity"])


def build_sector_map(codes: list[str]) -> dict:
    """real 模式下尝试用 akshare 概念/行业成分构造 code->板块。

    网络不可用或接口变动时返回空 dict,上游自动降级。
    这里只取「行业板块」一层,够用且调用量小。
    """
    mapping: dict[str, str] = {}
    try:
        import akshare as ak
        boards = ak.stock_board_industry_name_em()
        # 控制调用量:只取涨幅靠前的若干活跃板块
        boards = boards.sort_values("涨跌幅", ascending=False).head(15)
        wanted = set(str(c) for c in codes)
        for name in boards["板块名称"]:
            try:
                cons = ak.stock_board_industry_cons_em(symbol=name)
                for code in cons["代码"].astype(str):
                    if code in wanted and code not in mapping:
                        mapping[code] = name
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return {}
    return mapping
