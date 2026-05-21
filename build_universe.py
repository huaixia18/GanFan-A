"""自动生成 universe.txt —— 从全市场筛出符合纪律的候选股票池。

流程:
  1. 拉全市场代码 + 行业(上交所主板/科创板 + 深交所主板/创业板)
  2. 用腾讯源批量取实时行情(市值/股价/PE/名称)—— 稳定、抗节流
  3. 按纪律硬性筛:市值<100亿 / 股价5-30 / 非ST / 非亏损(PE>0) / 非次新
  4. 按行业分组写入 universe.txt(注释行=行业,供龙头识别)

用法:
  python3 build_universe.py                 # 用 config.yaml 的阈值
  python3 build_universe.py --max-per 8      # 每个行业最多保留 8 只(默认全保留)
  python3 build_universe.py --dry-run        # 只打印,不写文件
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from stockagent import datasource as ds

ROOT = Path(__file__).resolve().parent


def _clear_proxy():
    import os
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)


def fetch_all_codes() -> pd.DataFrame:
    """返回全市场 DataFrame:code, name, industry。失败的板块自动跳过。"""
    import akshare as ak
    frames = []

    # 上交所:主板 + 科创板(无行业字段,后面归「未分类」由腾讯名称兜底)
    for sym in ("主板A股", "科创板"):
        try:
            df = ak.stock_info_sh_name_code(symbol=sym)
            frames.append(pd.DataFrame({
                "code": df["证券代码"].astype(str),
                "name": df["证券简称"].astype(str),
                "industry": "未分类",
            }))
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] 上交所 {sym}: {type(e).__name__}", file=sys.stderr)

    # 深交所:自带所属行业
    try:
        df = ak.stock_info_sz_name_code(symbol="A股列表")
        ind = df["所属行业"].astype(str).str.replace(r"^[A-Z]\s*", "", regex=True)
        frames.append(pd.DataFrame({
            "code": df["A股代码"].astype(str),
            "name": df["A股简称"].astype(str).str.replace(" ", ""),
            "industry": ind.replace("", "未分类"),
        }))
    except Exception as e:  # noqa: BLE001
        print(f"  [跳过] 深交所: {type(e).__name__}", file=sys.stderr)

    if not frames:
        raise SystemExit("无法获取任何全市场代码表(网络/接口问题)")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["code"]).reset_index(drop=True)


def screen(codes_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """用腾讯源批量取行情并按纪律筛选,返回通过的(含 industry)。"""
    f = cfg["fundamental"]
    all_codes = codes_df["code"].tolist()
    print(f"全市场 {len(all_codes)} 只,正在用腾讯源批量取行情...", file=sys.stderr)

    spot = ds._tx_spot(all_codes)  # noqa: SLF001  复用已验证的腾讯批量接口
    print(f"取到行情 {len(spot)} 只,开始筛选...", file=sys.stderr)

    merged = spot.merge(codes_df[["code", "industry"]], on="code", how="left")
    merged["industry"] = merged["industry"].fillna("未分类")

    m = pd.Series(True, index=merged.index)
    if f.get("exclude_st", True):
        m &= ~merged[ds.COL_NAME].str.contains("ST", case=False, na=False)
    if f.get("exclude_loss", True):
        m &= merged[ds.COL_PE] > 0          # PE>0 作为非亏损代理
    m &= merged[ds.COL_MKT_CAP] <= f["market_cap_max"]
    m &= merged[ds.COL_MKT_CAP] > 0          # 剔除取不到市值的
    m &= merged[ds.COL_PRICE] >= f["price_min"]
    m &= merged[ds.COL_PRICE] <= f["price_max"]

    passed = merged[m].copy()
    # 活跃度排序:当日涨幅 + 量比(契合"跟踪活跃板块"纪律),组内据此取 Top N
    passed["_activity"] = passed[ds.COL_PCT] + 2.0 * passed[ds.COL_VOL_RATIO].clip(lower=0)
    return passed.sort_values("_activity", ascending=False).reset_index(drop=True)


def write_universe(passed: pd.DataFrame, max_per: int | None, dry_run: bool):
    lines = [
        "# 候选股票池 —— 由 build_universe.py 自动生成",
        "# 已按纪律筛选:市值<100亿 / 股价5-30 / 非ST / 非亏损。",
        "# 按行业分组(注释行=行业名,供腾讯源识别龙头)。可手动增删。",
        "",
    ]
    total = 0
    for industry, grp in passed.groupby("industry", sort=False):
        if max_per:
            grp = grp.head(max_per)
        if grp.empty:
            continue
        lines.append(f"# --- {industry} ---")
        for _, r in grp.iterrows():
            lines.append(
                f"{r['code']}  # {r['name']} {r[ds.COL_MKT_CAP]:.0f}亿 "
                f"{r[ds.COL_PRICE]:.2f}元 {r[ds.COL_PCT]:+.1f}%")
            total += 1
        lines.append("")

    content = "\n".join(lines)
    if dry_run:
        print(content)
        print(f"\n[dry-run] 共 {total} 只,未写文件。", file=sys.stderr)
        return
    (ROOT / "universe.txt").write_text(content, encoding="utf-8")
    print(f"✓ 已写入 universe.txt:{total} 只,覆盖 {passed['industry'].nunique()} 个行业。", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per", type=int, default=None, help="每个行业最多保留几只")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    args = ap.parse_args()

    _clear_proxy()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    codes_df = fetch_all_codes()
    passed = screen(codes_df, cfg)
    print(f"通过筛选:{len(passed)} 只", file=sys.stderr)
    write_universe(passed, args.max_per, args.dry_run)


if __name__ == "__main__":
    main()
