"""数据源层。

真实源优先级:腾讯(qt.gtimg.cn)→ 东方财富(akshare)。
腾讯源在国内网络几乎总是可达且抗节流;东方财富接口在部分网络被阻断;
两者都失败抛 DataUnavailable(不再造假数据)。

对外暴露:
  - spot():       现货快照(代码/名称/价格/涨跌幅/量比/换手/振幅/市值/PE 等)
  - daily(code):  单只前复权日线(算趋势用)
所有字段统一成内部规范列名,下游模块不直接接触原始字段。

注意:腾讯没有"全市场列表"接口,因此 spot() 针对 universe.txt 中的候选
代码池批量取数(自选池会并入)。东方财富源可取全市场。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = ROOT / "universe.txt"

_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/120.0.0.0 Safari/537.36")


def _no_proxy_session() -> requests.Session:
    """构造一个绕过环境代理的会话。

    环境里残留的 HTTP_PROXY 可能指向已关闭的本地代理(导致请求全失败),
    这里显式禁用代理直连——A股数据源都是境内,无需代理。
    """
    s = requests.Session()
    s.trust_env = False  # 忽略 HTTP_PROXY/HTTPS_PROXY 等环境变量
    s.headers.update({"User-Agent": _BROWSER_UA})
    return s

# 内部统一列名(下游只认这些)
COL_CODE = "code"
COL_NAME = "name"
COL_PRICE = "price"
COL_PCT = "pct_chg"          # 当日涨跌幅 %
COL_VOL_RATIO = "vol_ratio"  # 量比
COL_TURNOVER = "turnover"    # 换手率 %
COL_AMPLITUDE = "amplitude"  # 振幅 %
COL_MKT_CAP = "mkt_cap"      # 总市值,亿元
COL_PE = "pe"                # 动态市盈率(<0 视为亏损)
COL_AMOUNT = "amount"        # 成交额,元


class DataUnavailable(Exception):
    pass


# --------------------------------------------------------------------------- #
# 候选代码池
# --------------------------------------------------------------------------- #
def load_universe(extra: list[str] | None = None) -> list[str]:
    return list(load_universe_with_sectors(extra)[0])


def load_universe_with_sectors(
    extra: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """解析 universe.txt:返回 (代码列表, code->板块)。

    板块来自形如 `# --- 白酒/消费 ---` 的注释行,其下的代码归入该板块。
    腾讯源没有板块接口,靠这个分组识别龙头。
    """
    codes: list[str] = []
    sectors: dict[str, str] = {}
    current = "未分类"
    if UNIVERSE_PATH.exists():
        for line in UNIVERSE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                label = line.lstrip("# -").rstrip(" -").strip()
                if label:
                    current = label
                continue
            # 剥离行内注释:`300831  # 派瑞股份 ...` → `300831`
            code = line.split("#", 1)[0].strip()
            if not code:
                continue
            codes.append(code)
            sectors.setdefault(code, current)
    for c in extra or []:
        c = str(c).strip()
        if c and c not in codes:
            codes.append(c)
            sectors.setdefault(c, "自选")
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out, sectors


def _tx_prefix(code: str) -> str:
    """腾讯需要市场前缀:沪市 sh / 深市 sz / 北交所 bj。"""
    code = str(code)
    if code.startswith(("60", "68", "5", "11", "90")):
        return "sh" + code
    if code.startswith(("0", "30", "20", "12", "15", "16", "18")):
        return "sz" + code
    if code.startswith(("4", "8", "92")):
        return "bj" + code
    return "sh" + code


# --------------------------------------------------------------------------- #
# 真实数据源 A:腾讯(首选,抗节流、境内直连)
# --------------------------------------------------------------------------- #
# qt.gtimg.cn 单条记录字段索引(0-based,GBK 编码,'~' 分隔)
_TX_I = {
    "name": 1, "code": 2, "price": 3, "prev_close": 4,
    "pct_chg": 32, "turnover": 38, "pe": 39,
    "vol_ratio": 43, "mkt_cap": 45, "amplitude": 49, "amount_wan": 37,
}


def _tx_spot(codes: list[str]) -> pd.DataFrame:
    """批量取腾讯现货行情。codes 为裸 6 位码。"""
    if not codes:
        raise DataUnavailable("候选代码池为空")
    sess = _no_proxy_session()
    rows = []
    # 腾讯单次请求可带很多代码,分批 60 个,稳妥
    for i in range(0, len(codes), 60):
        batch = codes[i:i + 60]
        q = ",".join(_tx_prefix(c) for c in batch)
        resp = sess.get(f"http://qt.gtimg.cn/q={q}", timeout=8)
        resp.encoding = "gbk"
        for raw_line in resp.text.strip().split("\n"):
            if '="' not in raw_line:
                continue
            payload = raw_line.split('="', 1)[1].rstrip('";')
            f = payload.split("~")
            if len(f) < 50 or not f[_TX_I["code"]]:
                continue
            try:
                rows.append({
                    COL_CODE: f[_TX_I["code"]],
                    COL_NAME: f[_TX_I["name"]].replace(" ", ""),
                    COL_PRICE: float(f[_TX_I["price"]] or 0),
                    COL_PCT: float(f[_TX_I["pct_chg"]] or 0),
                    COL_VOL_RATIO: float(f[_TX_I["vol_ratio"]] or 0),
                    COL_TURNOVER: float(f[_TX_I["turnover"]] or 0),
                    COL_AMPLITUDE: float(f[_TX_I["amplitude"]] or 0),
                    COL_MKT_CAP: float(f[_TX_I["mkt_cap"]] or 0),  # 已是亿元
                    COL_PE: float(f[_TX_I["pe"]] or 0),
                    COL_AMOUNT: float(f[_TX_I["amount_wan"]] or 0) * 1e4,  # 万元→元
                })
            except (ValueError, IndexError):
                continue
    if not rows:
        raise DataUnavailable("腾讯现货返回为空")
    return pd.DataFrame(rows)


# 日线缓存:同一交易日内复用,避免每次刷新都串行拉 100+ 只(慢)。
# 磁盘缓存到 .cache/daily/<date>/ ,跨日自动失效(目录名带日期)。
_CACHE_DIR = ROOT / ".cache" / "daily"
_daily_mem: dict[str, pd.DataFrame] = {}


def _today_key() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d")


def _tx_daily(code: str, days: int) -> pd.DataFrame:
    cache_key = f"{code}_{days}"
    if cache_key in _daily_mem:
        return _daily_mem[cache_key]

    day_dir = _CACHE_DIR / _today_key()
    cache_file = day_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        df = pd.read_pickle(cache_file)
        _daily_mem[cache_key] = df
        return df

    sess = _no_proxy_session()
    sym = _tx_prefix(code)
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
           f"?param={sym},day,,,{max(days, 60)},qfq")
    resp = sess.get(url, timeout=10, allow_redirects=True)
    data = resp.json()["data"][sym]
    arr = data.get("qfqday") or data.get("day") or []
    if not arr:
        raise DataUnavailable(f"腾讯日线为空: {code}")
    df = pd.DataFrame(arr)
    # 每条:[日期, 开, 收, 高, 低, 量]
    df = df.iloc[:, :6]
    df.columns = ["date", "open", "close", "high", "low", "vol"]
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    out = df[["date", "close", "vol"]].tail(days).reset_index(drop=True)

    day_dir.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache_file)
    _daily_mem[cache_key] = out
    return out


def _tx_minute(code: str) -> list[float]:
    """当日分时价序列(逐分钟收盘价),供盘中卖出规则判定。"""
    sess = _no_proxy_session()
    sym = _tx_prefix(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={sym}"
    resp = sess.get(url, timeout=8, allow_redirects=True)
    mins = resp.json()["data"][sym]["data"]["data"]
    prices = []
    for row in mins:
        parts = row.split()
        if len(parts) >= 2:
            try:
                prices.append(float(parts[1]))
            except ValueError:
                continue
    return prices


# 全市场代码表缓存(当日)
_codes_cache: tuple[str, pd.DataFrame] | None = None


def full_market_codes() -> pd.DataFrame:
    """全市场代码表:code, name, industry。

    走交易所官网(沪深),不经被阻断的东财;当日内存缓存。
    """
    global _codes_cache
    today = pd.Timestamp.now().strftime("%Y%m%d")
    if _codes_cache and _codes_cache[0] == today:
        return _codes_cache[1]

    ak = _ak()
    frames = []
    for sym in ("主板A股", "科创板"):
        try:
            df = ak.stock_info_sh_name_code(symbol=sym)
            frames.append(pd.DataFrame({
                "code": df["证券代码"].astype(str),
                "name": df["证券简称"].astype(str),
                "industry": "未分类",
            }))
        except Exception:  # noqa: BLE001
            pass
    try:
        df = ak.stock_info_sz_name_code(symbol="A股列表")
        ind = df["所属行业"].astype(str).str.replace(r"^[A-Z]\s*", "", regex=True)
        frames.append(pd.DataFrame({
            "code": df["A股代码"].astype(str),
            "name": df["A股简称"].astype(str).str.replace(" ", ""),
            "industry": ind.replace("", "未分类"),
        }))
    except Exception:  # noqa: BLE001
        pass

    if not frames:
        raise DataUnavailable("无法获取全市场代码表")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["code"]).reset_index(drop=True)
    _codes_cache = (today, out)
    return out


# --------------------------------------------------------------------------- #
# 真实数据源 B:东方财富(akshare,可取全市场;部分网络被阻断)
# --------------------------------------------------------------------------- #
def _ak():
    import akshare as ak
    return ak


def _real_spot() -> pd.DataFrame:
    ak = _ak()
    raw = ak.stock_zh_a_spot_em()
    # 东方财富现货接口的中文列名 → 内部列名
    rename = {
        "代码": COL_CODE,
        "名称": COL_NAME,
        "最新价": COL_PRICE,
        "涨跌幅": COL_PCT,
        "量比": COL_VOL_RATIO,
        "换手率": COL_TURNOVER,
        "振幅": COL_AMPLITUDE,
        "总市值": COL_MKT_CAP,
        "市盈率-动态": COL_PE,
        "成交额": COL_AMOUNT,
    }
    df = raw.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    df = df[keep].copy()
    # 总市值原始单位为元 → 转亿元
    if COL_MKT_CAP in df.columns:
        df[COL_MKT_CAP] = pd.to_numeric(df[COL_MKT_CAP], errors="coerce") / 1e8
    for c in [COL_PRICE, COL_PCT, COL_VOL_RATIO, COL_TURNOVER,
              COL_AMPLITUDE, COL_PE, COL_AMOUNT]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[COL_CODE, COL_PRICE]).reset_index(drop=True)


def _real_daily(code: str, days: int) -> pd.DataFrame:
    ak = _ak()
    raw = ak.stock_zh_a_hist(symbol=code, period="daily",
                             adjust="qfq")
    rename = {"日期": "date", "收盘": "close", "成交量": "vol", "成交额": "amount"}
    df = raw.rename(columns=rename)[["date", "close", "vol", "amount"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    return df.tail(days).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 统一入口:真实源自动降级 腾讯 → 东方财富
# --------------------------------------------------------------------------- #
@dataclass
class DataSource:
    universe: list[str] | None = None   # 腾讯源的候选代码池(裸6位)
    prefer: str = "tencent"             # tencent / eastmoney(首选真实源)
    full_market: bool = False           # True=腾讯全市场模式(自动拉交易所代码表)
    mode: str = "unknown"               # tencent / eastmoney
    industry_map: dict[str, str] | None = None   # 全市场模式下 code->行业
    _backend: str | None = None         # 锁定本次会话用哪个真实源,保证 spot/daily 一致

    def __post_init__(self):
        if self.universe is None and not self.full_market:
            self.universe = load_universe()

    def _ensure_full_market_codes(self):
        """全市场模式:从交易所代码表取全部代码 + 行业映射。"""
        codes_df = full_market_codes()
        self.universe = codes_df["code"].tolist()
        self.industry_map = dict(zip(codes_df["code"], codes_df["industry"]))

    # ---- spot ---- #
    def spot(self) -> pd.DataFrame:
        order = (["tencent", "eastmoney"] if self.prefer == "tencent"
                 else ["eastmoney", "tencent"])
        errors = []
        for backend in order:
            try:
                if backend == "tencent":
                    if self.full_market and not self.universe:
                        self._ensure_full_market_codes()
                    df = _tx_spot(self.universe)
                else:
                    df = _real_spot()   # 东财天然全市场
                self._backend = backend
                self.mode = backend
                return df
            except Exception as e:  # noqa: BLE001
                errors.append(f"{backend}: {type(e).__name__} {str(e)[:80]}")

        raise DataUnavailable("所有真实源失败 -> " + " | ".join(errors))

    def minute_prices(self, code: str) -> list[float]:
        """当日分时价序列(腾讯)。取不到返回空。"""
        if self._backend is None:
            return []
        try:
            return _tx_minute(code)
        except Exception:  # noqa: BLE001
            return []

    # ---- daily ---- #
    def daily(self, code: str, days: int = 60) -> pd.DataFrame:
        backend = self._backend or self.mode
        if backend == "tencent":
            df = _tx_daily(code, days)
        else:
            df = _real_daily(code, days)
        if df is None or df.empty:
            raise DataUnavailable("空日线")
        return df

    @property
    def is_real(self) -> bool:
        return self.mode in ("tencent", "eastmoney")
