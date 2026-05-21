"""账户风控状态机 —— 铁的纪律:单日回撤 -4.5% 强制休息 2 天。

状态持久化到 state.json:
  - watchlist:        自选股池(也可由 config 提供,这里允许运行时增删)
  - resting_until:    休息截止日期(含),期间只看不买
  - last_drawdown:    最近记录的单日回撤
  - history:          每日回撤记录,便于复盘

休息期间 status="RESTING",界面应禁用买入提示,只展示观察。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


@dataclass
class RiskState:
    resting_until: str | None = None   # ISO date 字符串
    last_drawdown: float | None = None
    history: list[dict] = field(default_factory=list)
    watchlist: list[str] = field(default_factory=list)
    # 持仓:每项 {code, cost, shares, buy_date}
    positions: list[dict] = field(default_factory=list)
    # 账户每日净值快照(复盘曲线):每项 {date, value, pnl, pnl_pct, day_pnl, day_pct}
    equity_history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "RiskState":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                resting_until=data.get("resting_until"),
                last_drawdown=data.get("last_drawdown"),
                history=data.get("history", []),
                watchlist=[str(c) for c in data.get("watchlist", [])],
                positions=data.get("positions", []),
                equity_history=data.get("equity_history", []),
            )
        return cls()

    # ---- 账户净值快照(复盘曲线) ---- #
    def snapshot_equity(self, value: float, pnl: float, pnl_pct: float,
                        day_pnl: float, day_pct: float, today: date | None = None):
        """记录当日账户净值。当日已有则更新(取最后一次),跨日则新增 —— 一天一条。"""
        d = (today or date.today()).isoformat()
        rec = {
            "date": d,
            "value": round(value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "day_pnl": round(day_pnl, 2),
            "day_pct": round(day_pct, 2),
        }
        if self.equity_history and self.equity_history[-1]["date"] == d:
            self.equity_history[-1] = rec
        else:
            self.equity_history.append(rec)

    # ---- 持仓增删 ---- #
    def add_position(self, code: str, cost: float, shares: int, buy_date: str | None = None):
        code = str(code).strip()
        # 同代码合并:按加权平均成本累加(顺势加仓后成本会变)
        for p in self.positions:
            if p["code"] == code:
                total_sh = p["shares"] + shares
                if total_sh > 0:
                    p["cost"] = round((p["cost"] * p["shares"] + cost * shares) / total_sh, 3)
                p["shares"] = total_sh
                return
        self.positions.append({
            "code": code,
            "cost": round(float(cost), 3),
            "shares": int(shares),
            "buy_date": buy_date or date.today().isoformat(),
        })

    def remove_position(self, code: str):
        code = str(code).strip()
        self.positions = [p for p in self.positions if p["code"] != code]

    def save(self, path: Path):
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- #
    def record_drawdown(self, drawdown_pct: float, threshold: float,
                        rest_days: int, today: date | None = None) -> str:
        """登记当日回撤;达到阈值则触发强制休息。返回状态字符串。"""
        today = today or date.today()
        self.last_drawdown = drawdown_pct
        triggered = drawdown_pct <= threshold
        if triggered:
            until = today + timedelta(days=rest_days)
            self.resting_until = until.isoformat()
        self.history.append({
            "date": today.isoformat(),
            "drawdown": drawdown_pct,
            "triggered_rest": triggered,
        })
        return self.status(today)

    def status(self, today: date | None = None) -> str:
        today = today or date.today()
        if self.resting_until:
            until = date.fromisoformat(self.resting_until)
            if today <= until:
                return "RESTING"
        return "ACTIVE"

    def can_trade(self, today: date | None = None) -> bool:
        return self.status(today) == "ACTIVE"

    def resting_days_left(self, today: date | None = None) -> int:
        today = today or date.today()
        if self.resting_until:
            until = date.fromisoformat(self.resting_until)
            return max((until - today).days + 1, 0) if today <= until else 0
        return 0
