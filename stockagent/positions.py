"""持仓分析 —— 浮盈亏 / 当日损益 / 止损 / 加仓提示 / 风控联动。

严格遵守纪律:
  - **不补仓**:持仓下跌/亏损时绝不提示加仓,只提醒"纪律:不补仓"。
  - 加仓提示**只在上升趋势 + 有量承接**时给(顺势不逆,称"加仓"非"补仓")。
  - 卖出/止损果断:分时急拉急跌或缓拉急跌 → 卖;趋势转坏(跌破均线)→ 止损。
  - 账户总浮盈亏汇总,当日总回撤达 -4.5% 联动强制休息。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import datasource as ds
from .scorer import compute_trend
from .signals import intraday_sell_signal


@dataclass
class PositionView:
    code: str
    name: str
    cost: float
    shares: int
    buy_date: str
    price: float
    market_value: float
    pnl: float            # 浮动盈亏(元)
    pnl_pct: float        # 浮动盈亏 %
    day_pnl: float        # 当日损益(元)
    day_pct: float        # 当日涨跌 %
    uptrend: bool
    alerts: list[str] = field(default_factory=list)   # 卖出/止损/加仓提示
    level: str = "ok"     # ok / warn(止损/卖出) / add(可加仓)


@dataclass
class PortfolioView:
    items: list[PositionView]
    total_cost: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    total_day_pnl: float
    total_day_pct: float        # 账户当日回撤/盈利 %(对总成本)
    drawdown_alert: bool        # 当日总回撤是否达到 -4.5% 阈值


def analyze(positions: list[dict], source: ds.DataSource, cfg: dict,
            with_intraday: bool = True) -> PortfolioView:
    if not positions:
        return PortfolioView([], 0, 0, 0, 0, 0, 0, False)

    spot = source.spot()  # 锁定真实源 / 也供取价
    spot_idx = spot.set_index(ds.COL_CODE)
    hist_days = cfg["runtime"]["hist_days"]
    rest_threshold = cfg["risk_control"]["max_daily_drawdown"]

    items: list[PositionView] = []
    total_cost = total_value = total_day_pnl = 0.0

    for p in positions:
        code = str(p["code"])
        cost = float(p["cost"])
        shares = int(p["shares"])
        if code not in spot_idx.index:
            # 不在当前 spot(全市场模式应都在;候选模式可能缺)→ 单独取
            try:
                one = ds._tx_spot([code]).set_index(ds.COL_CODE)  # noqa: SLF001
                row = one.loc[code]
            except Exception:  # noqa: BLE001
                items.append(PositionView(code, code, cost, shares, p.get("buy_date", ""),
                                          0, 0, 0, 0, 0, 0, False,
                                          ["无法获取行情"], "warn"))
                continue
        else:
            row = spot_idx.loc[code]

        price = float(row[ds.COL_PRICE])
        name = str(row[ds.COL_NAME])
        day_pct = float(row[ds.COL_PCT])
        prev_close = price / (1 + day_pct / 100) if day_pct != -100 else price

        mv = price * shares
        pnl = (price - cost) * shares
        pnl_pct = (price - cost) / cost * 100 if cost else 0
        day_pnl = (price - prev_close) * shares

        # 趋势(止损/加仓判定)
        daily = source.daily(code, hist_days)
        tm = compute_trend(daily)

        alerts: list[str] = []
        level = "ok"

        # 1) 分时卖出信号(果断卖出)
        if with_intraday:
            mins = source.minute_prices(code)
            if mins:
                trig, msg = intraday_sell_signal(mins)
                if trig:
                    alerts.append(f"卖出:{msg}")
                    level = "warn"

        # 2) 趋势转坏止损:跌破 20 日均线 / 趋势不再向上
        if not tm.uptrend or not tm.above_ma20:
            alerts.append("趋势转坏(跌破均线/上升趋势破坏)→ 考虑止损")
            level = "warn"

        # 3) 浮亏提醒 + 不补仓纪律
        if pnl_pct < 0:
            alerts.append(f"浮亏 {pnl_pct:.1f}% · 纪律:不补仓")

        # 4) 加仓提示:仅上升趋势 + 有量承接 + 当前未触发卖出/止损
        vr = float(row.get(ds.COL_VOL_RATIO, 0) or 0)
        min_vr = cfg["buy_signal"].get("min_volume_ratio", 1.0)
        if level == "ok" and tm.uptrend and tm.above_ma20 and vr >= min_vr and pnl_pct >= 0:
            alerts.append(f"顺势可加仓(上升趋势+量比{vr:.1f}承接)")
            level = "add"

        items.append(PositionView(
            code=code, name=name, cost=cost, shares=shares,
            buy_date=p.get("buy_date", ""), price=round(price, 2),
            market_value=round(mv, 2), pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 2),
            day_pnl=round(day_pnl, 2), day_pct=round(day_pct, 2),
            uptrend=bool(tm.uptrend), alerts=alerts, level=level,
        ))
        total_cost += cost * shares
        total_value += mv
        total_day_pnl += day_pnl

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0
    # 当日账户回撤 %:当日损益 / (今日开盘时市值≈昨收市值)
    prev_value = total_value - total_day_pnl
    total_day_pct = (total_day_pnl / prev_value * 100) if prev_value else 0
    drawdown_alert = bool(total_day_pct <= rest_threshold)

    return PortfolioView(
        items=items,
        total_cost=round(total_cost, 2),
        total_value=round(total_value, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 2),
        total_day_pnl=round(total_day_pnl, 2),
        total_day_pct=round(total_day_pct, 2),
        drawdown_alert=drawdown_alert,
    )
