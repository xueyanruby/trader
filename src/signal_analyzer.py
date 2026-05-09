"""
技術信號分析器：短線 / 長線買入信號識別

短線信號（三個 Lean 移植子信號組合）
--------------------------------------
移植自 QuantConnect Lean 的三個 Alpha Model：
  1. EMA Cross  ← Lean EmaCrossAlphaModel  ：快線上穿慢線（金叉）
  2. RSI        ← Lean RsiAlphaModel       ：RSI 跌入超賣區後回升
  3. MACD       ← Lean MacdAlphaModel      ：signal line 相對股價歸一化值 > 閾值

觸發模式（trigger_mode）：
  "any"      — 任意一個子信號觸發即發出買入信號（默認）
  "majority" — 超過半數（≥ 2/3）子信號同時觸發
  "all"      — 三個子信號全部觸發（最嚴格）

長線信號（MA 金叉 + 趨勢確認）
---------------------------------
  MA(fast) 上穿 MA(slow)，且收盤價高於 MA(trend)（趨勢確認）

信號結果格式
-----------
每個 symbol 返回 SignalResult：
  signal_type : "short" | "long" | "both" | None
  short_detail: dict（各子信號狀態 + 觸發原因）
  long_detail : dict（MA 值 + 觸發原因）
  price       : float（最新收盤價）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# 數據結構
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    symbol: str
    price: float
    signal_type: Optional[str] = None     # "short" | "long" | "both" | None
    short_fired: bool = False
    long_fired: bool = False
    short_detail: Dict[str, Any] = field(default_factory=dict)
    long_detail: Dict[str, Any] = field(default_factory=dict)

    def has_signal(self) -> bool:
        return self.short_fired or self.long_fired


# ---------------------------------------------------------------------------
# 底層指標計算（純 pandas，無外部依賴）
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI（與 Lean RsiAlphaModel 一致：com = period-1）"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - 100 / (1 + rs)


def _macd_signal_line(
    close: pd.Series,
    fast: int,
    slow: int,
    signal_period: int,
) -> pd.Series:
    """返回 MACD signal line（移植自 Lean MacdAlphaModel）"""
    macd_line = _ema(close, fast) - _ema(close, slow)
    return _ema(macd_line, signal_period)


def _cross_above(fast: pd.Series, slow: pd.Series, tolerance: int = 3) -> bool:
    """判斷 fast 是否在最近 tolerance 根 K 線內上穿 slow（金叉）"""
    if len(fast) < 2:
        return False
    window = min(tolerance + 1, len(fast))
    f = fast.iloc[-window:]
    s = slow.iloc[-window:]
    if float(f.iloc[-1]) <= float(s.iloc[-1]):
        return False
    return any(float(f.iloc[i]) <= float(s.iloc[i]) for i in range(len(f) - 1))


# ---------------------------------------------------------------------------
# 主分析器
# ---------------------------------------------------------------------------

class SignalAnalyzer:
    """
    對單只股票的日線收盤序列計算短線 / 長線買入信號。

    短線使用三個 Lean Alpha Model 移植子信號：EMA Cross、RSI、MACD。
    長線使用 MA 金叉 + 趨勢確認。

    Parameters
    ----------
    short_cfg : dict  config.yaml → watch.short_term_signal
    long_cfg  : dict  config.yaml → watch.long_term_signal
    """

    def __init__(
        self,
        short_cfg: Dict[str, Any],
        long_cfg: Dict[str, Any],
    ):
        # ── 短線：基礎開關 ──────────────────────────────────────────────
        self.short_enabled: bool = short_cfg.get("enabled", True)
        self.short_trigger_mode: str = short_cfg.get("trigger_mode", "any")
        # EMA Cross 子信號（移植自 Lean EmaCrossAlphaModel）
        self.ema_fast: int = int(short_cfg.get("ema_fast", 12))
        self.ema_slow: int = int(short_cfg.get("ema_slow", 26))
        self.short_cross_tol: int = int(short_cfg.get("cross_tolerance", 3))
        # RSI 子信號（移植自 Lean RsiAlphaModel）
        self.rsi_period: int = int(short_cfg.get("rsi_period", 14))
        self.rsi_oversold: float = float(short_cfg.get("rsi_oversold", 30))
        self.rsi_exit_oversold: float = float(short_cfg.get("rsi_exit_oversold", 35))
        # MACD 子信號（移植自 Lean MacdAlphaModel）
        self.macd_fast: int = int(short_cfg.get("macd_fast", 12))
        self.macd_slow: int = int(short_cfg.get("macd_slow", 26))
        self.macd_signal: int = int(short_cfg.get("macd_signal", 9))
        self.macd_threshold: float = float(short_cfg.get("macd_bounce_threshold", 0.01))
        self.short_lookback: int = int(short_cfg.get("lookback_days", 60))

        # ── 長線：MA 金叉 + 趨勢 ────────────────────────────────────────
        self.long_enabled: bool = long_cfg.get("enabled", True)
        self.ma_fast: int = int(long_cfg.get("ma_fast", 20))
        self.ma_slow: int = int(long_cfg.get("ma_slow", 50))
        self.ma_trend: int = int(long_cfg.get("ma_trend", 200))
        self.long_lookback: int = int(long_cfg.get("lookback_days", 250))
        self.long_cross_tol: int = int(long_cfg.get("cross_tolerance", 3))

    # ------------------------------------------------------------------
    # 公開接口
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, close: pd.Series) -> SignalResult:
        """
        分析單只股票收盤價序列，返回 SignalResult。

        Parameters
        ----------
        symbol : str
        close  : pd.Series（索引為日期，按升序排列，最新在末尾）
        """
        close = close.dropna().sort_index()
        if close.empty:
            return SignalResult(symbol=symbol, price=float("nan"))

        price = float(close.iloc[-1])
        result = SignalResult(symbol=symbol, price=price)

        if self.short_enabled:
            result.short_fired, result.short_detail = self._check_short(close, price)

        if self.long_enabled:
            result.long_fired, result.long_detail = self._check_long(close)

        if result.short_fired and result.long_fired:
            result.signal_type = "both"
        elif result.short_fired:
            result.signal_type = "short"
        elif result.long_fired:
            result.signal_type = "long"

        return result

    def analyze_batch(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.Series],
    ) -> List[SignalResult]:
        """批量分析，返回所有有信號的 SignalResult。"""
        results = []
        for sym in symbols:
            series = price_data.get(sym)
            if series is None or len(series) < 2:
                continue
            try:
                r = self.analyze(sym, series)
                if r.has_signal():
                    results.append(r)
            except Exception as exc:
                logger.debug(f"[SignalAnalyzer] {sym} 分析異常：{exc}")
        return results

    # ------------------------------------------------------------------
    # 短線邏輯（三個 Lean 移植子信號）
    # ------------------------------------------------------------------

    def _check_short(self, close: pd.Series, price: float) -> Tuple[bool, dict]:
        """
        運行三個子信號並按 trigger_mode 決定是否觸發。
        返回 (fired: bool, detail: dict)
        """
        min_bars = max(self.ema_slow, self.macd_slow + self.macd_signal, self.rsi_period * 3)
        if len(close) < min_bars:
            return False, {"skip_reason": f"數據不足（需 >= {min_bars} 日）"}

        # ── 子信號 1：EMA Cross（Lean EmaCrossAlphaModel）──────────────
        ema_f = _ema(close, self.ema_fast)
        ema_s = _ema(close, self.ema_slow)
        ema_cross_buy = _cross_above(ema_f, ema_s, self.short_cross_tol)
        ema_detail = {
            f"EMA{self.ema_fast}": round(float(ema_f.iloc[-1]), 4),
            f"EMA{self.ema_slow}": round(float(ema_s.iloc[-1]), 4),
            "EMA金叉(Lean EmaCross)": ema_cross_buy,
        }

        # ── 子信號 2：RSI 超賣回升（Lean RsiAlphaModel）─────────────────
        rsi = _wilder_rsi(close, self.rsi_period)
        curr_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2]) if len(rsi) >= 2 else curr_rsi
        # 狀態轉換：前一根在超賣區，或現在仍在超賣區（等同 Lean 的 TRIPPED_LOW 狀態）
        rsi_was_oversold = prev_rsi < self.rsi_oversold
        rsi_oversold_now = curr_rsi < self.rsi_oversold
        # 回升觸發：前一根超賣 + 本根回升至 exit 閾值之上（Lean 的 TRIPPED_LOW→MIDDLE 轉換）
        rsi_reversal = rsi_was_oversold and curr_rsi >= self.rsi_exit_oversold
        rsi_buy = rsi_oversold_now or rsi_reversal
        rsi_detail = {
            "RSI(Wilder)": round(curr_rsi, 2),
            f"超賣(<{self.rsi_oversold})": rsi_oversold_now,
            "超賣回升信號(Lean RSI)": rsi_reversal,
        }

        # ── 子信號 3：MACD Signal Line（Lean MacdAlphaModel）────────────
        sig_line = _macd_signal_line(close, self.macd_fast, self.macd_slow, self.macd_signal)
        normalized = float(sig_line.iloc[-1]) / price if price != 0 else 0.0
        macd_buy = normalized > self.macd_threshold
        macd_detail = {
            f"MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})Signal": round(float(sig_line.iloc[-1]), 6),
            f"Normalized(>{self.macd_threshold})": round(normalized, 6),
            "MACD買入(Lean MACD)": macd_buy,
        }

        # ── 合併子信號 ──────────────────────────────────────────────────
        sub_signals = [ema_cross_buy, rsi_buy, macd_buy]
        n_fired = sum(sub_signals)
        mode = self.short_trigger_mode

        if mode == "all":
            fired = all(sub_signals)
            trigger_desc = "全部三個子信號"
        elif mode == "majority":
            fired = n_fired >= 2
            trigger_desc = f"多數票（{n_fired}/3）"
        else:    # "any"
            fired = any(sub_signals)
            active = [name for name, v in [("EMA金叉", ema_cross_buy), ("RSI超賣", rsi_buy), ("MACD", macd_buy)] if v]
            trigger_desc = "+".join(active) if active else ""

        detail = {
            **ema_detail,
            **rsi_detail,
            **macd_detail,
            "觸發模式": mode,
            "已觸發子信號": f"{n_fired}/3",
            "觸發原因": trigger_desc,
        }
        return fired, detail

    # ------------------------------------------------------------------
    # 長線邏輯（MA 金叉 + 趨勢確認）
    # ------------------------------------------------------------------

    def _check_long(self, close: pd.Series) -> Tuple[bool, dict]:
        """返回 (fired: bool, detail: dict)"""
        if len(close) < self.ma_trend:
            return False, {"skip_reason": f"數據不足（需 >= {self.ma_trend} 日）"}

        ma_f = _sma(close, self.ma_fast)
        ma_s = _sma(close, self.ma_slow)
        ma_t = _sma(close, self.ma_trend)

        ma_cross = _cross_above(ma_f, ma_s, self.long_cross_tol)
        above_trend = float(close.iloc[-1]) > float(ma_t.iloc[-1])

        detail = {
            f"MA{self.ma_fast}": round(float(ma_f.iloc[-1]), 4),
            f"MA{self.ma_slow}": round(float(ma_s.iloc[-1]), 4),
            f"MA{self.ma_trend}": round(float(ma_t.iloc[-1]), 4),
            "MA金叉": ma_cross,
            f"價格>MA{self.ma_trend}(趨勢確認)": above_trend,
        }

        fired = ma_cross and above_trend
        if fired:
            detail["觸發原因"] = f"MA{self.ma_fast}/MA{self.ma_slow} 金叉 + MA{self.ma_trend} 趨勢確認"

        return fired, detail
