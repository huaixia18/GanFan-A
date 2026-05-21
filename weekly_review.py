"""每周自动复盘进化 —— 供定时任务调用。

流程:回收到期预测的真实 N 日收益 → 评估当前权重 → 在安全闸内进化/回滚 → 打印报告。
不调参时(样本不足 / 无更优)也会安全退出,绝不破坏纪律红线。

手动跑:   python3 weekly_review.py
定时(每周一收盘后,示例 crontab):
   30 15 * * 1  cd /path/to/选股 && /usr/bin/python3 weekly_review.py >> review.log 2>&1
"""
from __future__ import annotations

from datetime import datetime

from stockagent import datasource as ds, evolve
from stockagent.pipeline import load_config


def main():
    cfg = load_config()
    print(f"=== 每周复盘 {datetime.now():%Y-%m-%d %H:%M} ===")

    source = ds.DataSource(prefer=cfg["runtime"].get("prefer_source", "tencent"))
    source.spot()  # 锁定真实源

    newly = evolve.verify_returns(source)
    print(f"新回收已验证样本: {newly} 条")

    report = evolve.evolve(cfg["score_weights"])
    print(f"样本: {report['samples']}/{report['min_samples']} · 动作: {report['action']}")
    print(f"说明: {report['note']}")
    if report.get("new_weights"):
        print(f"新权重: {report['new_weights']}")
        print(f"新指标: {report.get('new_metrics')}")


if __name__ == "__main__":
    main()
