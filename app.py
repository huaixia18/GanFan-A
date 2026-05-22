"""选股 Agent 简单界面。

  GET  /              选股看板:风控状态灯 + 排序选股表 + 买入判定
  POST /run           重新跑一次流程
  POST /drawdown      登记当日回撤(触发 -4.5% 强制休息)
  POST /watchlist     增删自选池
  POST /sell-check    分时卖出信号自测(粘贴分时价序列)
  GET  /api/intraday  盘中监控:自选池每只的实时分时 + 卖出信号(JSON,前端轮询)
"""
from __future__ import annotations

from datetime import date, datetime

from flask import Flask, jsonify, redirect, render_template, request, url_for

from stockagent import datasource as ds
from stockagent import positions as pos
from stockagent.pipeline import STATE_PATH, load_config, run
from stockagent.risk import RiskState
from stockagent.signals import intraday_sell_signal

app = Flask(__name__)


def _watchlist_codes(cfg) -> list[str]:
    state = RiskState.load(STATE_PATH)
    return list({*(str(c) for c in cfg.get("watchlist", []) or []),
                 *(str(c) for c in state.watchlist)})


@app.route("/")
def index():
    cfg = load_config()
    result = run(cfg)
    state = RiskState.load(STATE_PATH)
    # 当前生效的排除板块:state 覆盖优先,否则 config 默认
    eff_boards = (state.exclude_boards if state.exclude_boards is not None
                  else cfg["fundamental"].get("exclude_boards", []))
    return render_template("index.html", r=result, cfg=cfg,
                           positions=state.positions, exclude_boards=eff_boards)


@app.route("/run", methods=["POST"])
def rerun():
    return redirect(url_for("index"))


@app.route("/drawdown", methods=["POST"])
def drawdown():
    cfg = load_config()
    try:
        dd = float(request.form.get("drawdown", "0"))
    except ValueError:
        dd = 0.0
    state = RiskState.load(STATE_PATH)
    rc = cfg["risk_control"]
    state.record_drawdown(dd, rc["max_daily_drawdown"], rc["rest_days"], date.today())
    state.save(STATE_PATH)
    return redirect(url_for("index"))


@app.route("/watchlist", methods=["POST"])
def watchlist():
    state = RiskState.load(STATE_PATH)
    code = (request.form.get("code") or "").strip()
    action = request.form.get("action")
    if code:
        if action == "add" and code not in state.watchlist:
            state.watchlist.append(code)
        elif action == "remove" and code in state.watchlist:
            state.watchlist.remove(code)
        state.save(STATE_PATH)
    return redirect(url_for("index"))


@app.route("/boards", methods=["POST"])
def boards():
    """设置排除板块。表单提交勾选的板块代码(chinext/star/bse),未勾的即不排除。"""
    state = RiskState.load(STATE_PATH)
    valid = {"chinext", "star", "bse"}
    selected = [b for b in request.form.getlist("exclude") if b in valid]
    state.exclude_boards = selected  # 即使空列表也存(表示用户明确不排除)
    state.save(STATE_PATH)
    return redirect(url_for("index"))


@app.route("/reset-rest", methods=["POST"])
def reset_rest():
    state = RiskState.load(STATE_PATH)
    state.resting_until = None
    state.save(STATE_PATH)
    return redirect(url_for("index"))


@app.route("/sell-check", methods=["POST"])
def sell_check():
    raw = request.form.get("prices", "")
    try:
        prices = [float(x) for x in raw.replace(",", " ").split() if x.strip()]
    except ValueError:
        prices = []
    triggered, msg = intraday_sell_signal(prices) if len(prices) >= 4 else (False, "请至少输入4个价格")
    cfg = load_config()
    result = run(cfg)
    return render_template("index.html", r=result, cfg=cfg,
                           sell_result={"triggered": triggered, "msg": msg, "prices": raw})


@app.route("/position", methods=["POST"])
def position():
    state = RiskState.load(STATE_PATH)
    action = request.form.get("action")
    code = (request.form.get("code") or "").strip()
    if action == "add" and code:
        try:
            cost = float(request.form.get("cost", "0"))
            shares = int(float(request.form.get("shares", "0")))
        except ValueError:
            cost, shares = 0.0, 0
        if cost > 0 and shares > 0:
            state.add_position(code, cost, shares,
                               request.form.get("buy_date") or None)
    elif action == "remove" and code:
        state.remove_position(code)
    state.save(STATE_PATH)
    return redirect(url_for("index"))


@app.route("/api/positions")
def api_positions():
    """持仓监控:实时浮盈亏 + 卖出/止损/加仓提示 + 账户回撤联动风控(JSON,前端轮询)。"""
    cfg = load_config()
    state = RiskState.load(STATE_PATH)
    if not state.positions:
        return jsonify({"items": [], "empty": True,
                        "time": datetime.now().strftime("%H:%M:%S")})

    source = ds.DataSource(prefer=cfg["runtime"].get("prefer_source", "tencent"))
    pv = pos.analyze(state.positions, source, cfg, with_intraday=True)

    # 账户当日总回撤达阈值 → 自动登记并触发强制休息(铁律联动)
    auto_rested = False
    if pv.drawdown_alert and state.status() == "ACTIVE":
        rc = cfg["risk_control"]
        state.record_drawdown(pv.total_day_pct, rc["max_daily_drawdown"],
                              rc["rest_days"], date.today())
        auto_rested = True

    # 记录当日账户净值快照(复盘曲线),仅真实源
    if source.is_real:
        state.snapshot_equity(pv.total_value, pv.total_pnl, pv.total_pnl_pct,
                              pv.total_day_pnl, pv.total_day_pct, date.today())
    state.save(STATE_PATH)

    return jsonify({
        "empty": False,
        "mode": source.mode,
        "time": datetime.now().strftime("%H:%M:%S"),
        "account": {
            "total_value": pv.total_value,
            "total_pnl": pv.total_pnl,
            "total_pnl_pct": pv.total_pnl_pct,
            "total_day_pnl": pv.total_day_pnl,
            "total_day_pct": pv.total_day_pct,
            "drawdown_alert": pv.drawdown_alert,
            "auto_rested": auto_rested,
        },
        "items": [vars(it) for it in pv.items],
    })


@app.route("/api/evolve/status")
def api_evolve_status():
    """进化看板:当前权重版本 / 样本数 / IC / 历次调整。"""
    from stockagent import evolve
    st = evolve.status()
    cfg = load_config()
    st["config_weights"] = cfg["score_weights"]
    st["active_weights"] = evolve.active_weights(cfg["score_weights"])
    return jsonify(st)


@app.route("/api/evolve/run", methods=["POST"])
def api_evolve_run():
    """手动触发一次复盘进化:回收收益 → 评估 → 进化/回滚。"""
    from stockagent import evolve
    cfg = load_config()
    source = ds.DataSource(prefer=cfg["runtime"].get("prefer_source", "tencent"))
    source.spot()  # 锁定真实源
    verified = evolve.verify_returns(source)
    report = evolve.evolve(cfg["score_weights"])
    report["newly_verified"] = verified
    return jsonify(report)


@app.route("/api/equity")
def api_equity():
    """账户每日净值快照历史(复盘曲线)。"""
    state = RiskState.load(STATE_PATH)
    return jsonify({"history": state.equity_history})


@app.route("/api/intraday")
def api_intraday():
    """盘中监控:对自选池每只取实时分时,跑卖出规则。前端定时轮询。"""
    cfg = load_config()
    codes = _watchlist_codes(cfg)
    source = ds.DataSource(prefer=cfg["runtime"].get("prefer_source", "tencent"))
    source.spot()  # 锁定真实源后端(决定 minute_prices 是否可用)

    items = []
    for code in codes:
        prices = source.minute_prices(code)
        if not prices:
            items.append({"code": code, "ok": False, "msg": "无分时数据(休市/数据源不可用)"})
            continue
        triggered, msg = intraday_sell_signal(prices)
        last = prices[-1]
        first = prices[0]
        items.append({
            "code": code,
            "ok": True,
            "last": round(last, 2),
            "pct_from_open": round((last - first) / first * 100, 2) if first else 0,
            "points": len(prices),
            "triggered": triggered,
            "msg": msg,
        })
    return jsonify({
        "mode": source.mode,
        "time": datetime.now().strftime("%H:%M:%S"),
        "watchlist_size": len(codes),
        "items": items,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
