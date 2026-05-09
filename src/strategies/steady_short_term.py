"""
穩健短線策略（Steady Short-term）

適合持有 1～3 天，不追高、不打板。三個市場（A 股 / 港股 / 美股）通用，
僅依賴 OHLCV 數據，可直接接入現有回測引擎。

入選條件（七個條件必須全部滿足）
-----------------------------------
1. 均線多頭排列：MA5 > MA10 > MA20
2. 現價站在 MA5 / MA20 上方
3. 今日成交量 > 5 日均量（溫和放量）
4. 成交量未超 2 倍均量（避高位巨量）
5. 近 3 天慢漲：每日收盤溫和上漲，且無長上影線（上影線佔振幅比 < 40%）
6. 未連漲 4 天以上（避免追頂）
7. 當日漲幅：1% ～ 4%（不追漲停）

口訣：均線多頭 + 站在線上 + 溫和放量 + 小漲不追高

Config keys（config.yaml strategy.steady_short_term 節）
---------------------------------------------------------
ma_periods                  list[int]  均線週期列表              default [5, 10, 20]
vol_ratio_min               float      量比下限                  default 1.0
vol_ratio_max               float      量比上限                  default 2.0
gain_min                    float      最低漲幅（小數）           default 0.01
gain_max                    float      最高漲幅（小數）           default 0.04
max_upper_wick_ratio        float      上影線佔振幅比上限         default 0.40
lookback_days_pattern       int        價格形態回望天數           default 3
max_consecutive_up_days     int        連漲天數上限（含）         default 3
enable_pattern_check        bool       是否啟用近N天形態+上影線   default True
                                         CN=True（A股完整7條件）
                                         HK/US=False（4條件精簡版）
enable_consecutive_up_check bool       是否啟用連漲天數限制       default True
                                         CN=True  HK/US=False
top_n                       int        最多選股數（0 = 不限）     default 20
rebalance_freq              str        "D" | "W"                 default "D"
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy


@register_strategy("steady_short_term")
class SteadyShortTermStrategy(AbstractStrategy):
    """
    穩健短線策略 — A 股 / 港股 / 美股 通用版本。

    generate_signals() 只依賴 OHLCV，適用於所有市場。
    返回 +1 表示「今日收盤後可考慮買入」。
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ma_periods: List[int] = self._cfg("ma_periods", [5, 10, 20])
        vol_ratio_min: float = self._cfg("vol_ratio_min", 1.0)
        vol_ratio_max: float = self._cfg("vol_ratio_max", 2.0)
        gain_min: float = self._cfg("gain_min", 0.01)
        gain_max: float = self._cfg("gain_max", 0.04)
        max_wick_ratio: float = self._cfg("max_upper_wick_ratio", 0.40)
        lookback: int = self._cfg("lookback_days_pattern", 3)
        max_consec: int = self._cfg("max_consecutive_up_days", 3)
        enable_pattern: bool = self._cfg("enable_pattern_check", True)
        enable_consec: bool = self._cfg("enable_consecutive_up_check", True)
        top_n: int = self._cfg("top_n", 20)

        ma_periods_sorted = sorted(ma_periods)
        min_period = ma_periods_sorted[-1]  # 最長均線週期

        # 需要至少 min_period + lookback + 1 根 K 線
        required = min_period + lookback + 1
        if not isinstance(data.columns, pd.MultiIndex):
            logger.warning("SteadyShortTerm: data 不是 MultiIndex，跳過")
            return pd.Series(dtype=float)

        close = Preprocessor.get_close(data)
        if len(close) < required:
            logger.warning(
                f"SteadyShortTerm: 歷史不足（需 {required} 天，實有 {len(close)} 天）"
            )
            return pd.Series(dtype=float)

        volume = Preprocessor.get_volume(data)

        # 嘗試獲取 high / low / open（用於上影線計算）
        try:
            high = data.xs("high", axis=1, level="field")
            low = data.xs("low", axis=1, level="field")
            open_ = data.xs("open", axis=1, level="field")
            has_hlc = True
        except KeyError:
            has_hlc = False
            logger.debug("SteadyShortTerm: 缺少 high/low/open 欄位，跳過上影線檢查")

        signals: Dict[str, int] = {}

        for sym in close.columns:
            c = close[sym].dropna()
            v = volume[sym].dropna() if sym in volume.columns else pd.Series(dtype=float)

            if len(c) < required or len(v) < 6:
                continue

            # ── 條件 1 & 2：均線結構 ────────────────────────────────────
            mas = {p: float(c.iloc[-p:].mean()) for p in ma_periods_sorted}
            price_now = float(c.iloc[-1])

            # 多頭排列：短 > 中 > 長
            ma_vals = [mas[p] for p in ma_periods_sorted]  # 升序週期 = [5,10,20]
            if not all(ma_vals[i] > ma_vals[i + 1] for i in range(len(ma_vals) - 1)):
                continue

            # 現價站在最短均線（MA5）及最長均線（MA20）上方
            if price_now <= mas[ma_periods_sorted[0]] or price_now <= mas[ma_periods_sorted[-1]]:
                continue

            # ── 條件 3 & 4：量比 ─────────────────────────────────────────
            avg_vol_5 = float(v.iloc[-6:-1].mean())  # 昨天為基準的 5 日均量
            vol_today = float(v.iloc[-1])
            if avg_vol_5 <= 0:
                continue
            vol_ratio = vol_today / avg_vol_5
            if not (vol_ratio_min <= vol_ratio < vol_ratio_max):
                continue

            # ── 條件 5：近 lookback 天慢漲 + 無長上影線（可選）──────────
            # CN 版開啟（enable_pattern_check=True），HK/US 版可關閉
            if enable_pattern:
                recent_close = c.iloc[-(lookback + 1):]
                daily_gains = recent_close.pct_change().iloc[1:]

                # 每天須溫和上漲（> 0）且漲幅不超過 5%（排除連板）
                if not all(0 < g <= 0.05 for g in daily_gains):
                    continue

                # 上影線檢查（需要 high/low/open 數據）
                if has_hlc and sym in high.columns:
                    h = high[sym].dropna()
                    lo = low[sym].dropna()
                    o = open_[sym].dropna()
                    if len(h) >= lookback:
                        wick_ok = True
                        for i in range(-lookback, 0):
                            try:
                                h_val = float(h.iloc[i])
                                lo_val = float(lo.iloc[i])
                                o_val = float(o.iloc[i])
                                c_val = float(c.iloc[i])
                            except (IndexError, TypeError):
                                break
                            upper_wick = h_val - max(o_val, c_val)
                            candle_range = h_val - lo_val
                            if candle_range > 0:
                                wick_ratio = upper_wick / candle_range
                                if wick_ratio > max_wick_ratio:
                                    wick_ok = False
                                    break
                        if not wick_ok:
                            continue

            # ── 條件 6：未連漲超過 max_consec 天（可選）────────────────
            # CN 版開啟（enable_consecutive_up_check=True），HK/US 版可關閉
            if enable_consec:
                consec_up = self._count_consecutive_up(c)
                if consec_up > max_consec:
                    continue

            # ── 條件 7：當日漲幅 1%～4% ──────────────────────────────────
            if len(c) < 2:
                continue
            today_gain = (float(c.iloc[-1]) - float(c.iloc[-2])) / float(c.iloc[-2])
            if not (gain_min <= today_gain <= gain_max):
                continue

            signals[sym] = 1

        if not signals:
            return pd.Series(dtype=float)

        result = pd.Series(signals, dtype=int)

        # 按「量比適中 + 漲幅較小」排序（選更保守的票），再取 top_n
        if top_n > 0 and len(result) > top_n:
            buy_syms = result[result == 1].index.tolist()
            # 用漲幅由小到大排序（更保守、追高風險低）
            gains = {}
            for sym in buy_syms:
                c = close[sym].dropna()
                if len(c) >= 2:
                    gains[sym] = (float(c.iloc[-1]) - float(c.iloc[-2])) / float(c.iloc[-2])
            gain_series = pd.Series(gains)
            top_syms = gain_series.nsmallest(top_n).index.tolist()
            result = result[result.index.isin(top_syms)]

        logger.info(
            f"SteadyShortTerm: {len(result)} 只股票通過全部 7 個條件"
        )
        return result[result != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "steady_short_term",
            "ma_periods": self._cfg("ma_periods", [5, 10, 20]),
            "vol_ratio_min": self._cfg("vol_ratio_min", 1.0),
            "vol_ratio_max": self._cfg("vol_ratio_max", 2.0),
            "gain_min": self._cfg("gain_min", 0.01),
            "gain_max": self._cfg("gain_max", 0.04),
            "max_upper_wick_ratio": self._cfg("max_upper_wick_ratio", 0.40),
            "lookback_days_pattern": self._cfg("lookback_days_pattern", 3),
            "max_consecutive_up_days": self._cfg("max_consecutive_up_days", 3),
            "enable_pattern_check": self._cfg("enable_pattern_check", True),
            "enable_consecutive_up_check": self._cfg("enable_consecutive_up_check", True),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "D"),
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _count_consecutive_up(close: pd.Series) -> int:
        """
        從最後一天向前計算連漲天數。
        例：[100, 101, 102, 103] → 最後 3 天連漲，返回 3。
        """
        n = len(close)
        if n < 2:
            return 0
        count = 0
        for i in range(n - 1, 0, -1):
            if float(close.iloc[i]) > float(close.iloc[i - 1]):
                count += 1
            else:
                break
        return count
