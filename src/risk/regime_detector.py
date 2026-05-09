"""
市況狀態機（Regime Detector）

根據指數價格趨勢和實時波動率，將市場劃分為三種狀態：
  - bull    (牛市)   : 快均線 > 慢均線  且  波動率 < 閾值         → 最大倉位 40%
  - neutral (中性)   : 介於牛熊之間                               → 最大倉位 20%
  - bear    (熊市)   : 快均線 < 慢均線  或  波動率 > 1.5× 閾值    → 最大倉位 5%

狀態轉換帶有滯後性（防止頻繁切換），可通過 `hysteresis_bars` 控制。

使用方式
--------
detector = RegimeDetector(fast_period=20, slow_period=60)
regime, max_exposure = detector.detect(index_prices_series, realized_vol=0.18)
# regime  ∈ {"bull", "neutral", "bear"}
# max_exposure ∈ {0.40, 0.20, 0.05}
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Tuple

import pandas as pd
from loguru import logger


class RegimeDetector:
    """
    市況狀態機：根據 MA 趨勢 + 波動率動態確定最大組合倉位上限。

    Parameters
    ----------
    fast_period     : 快均線週期（交易日，默認 20）
    slow_period     : 慢均線週期（交易日，默認 60）
    vol_threshold   : 年化波動率閾值（默認 0.25 = 25%）
    hysteresis_bars : 狀態切換所需連續確認 K 線數（防抖，默認 3）
    exposure_bull   : 牛市最大組合倉位（默認 0.40）
    exposure_neutral: 中性最大組合倉位（默認 0.20）
    exposure_bear   : 熊市最大組合倉位（默認 0.05）
    """

    REGIME_LABELS = ("bull", "neutral", "bear")

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 60,
        vol_threshold: float = 0.25,
        hysteresis_bars: int = 3,
        exposure_bull: float = 0.40,
        exposure_neutral: float = 0.20,
        exposure_bear: float = 0.05,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_threshold = vol_threshold
        self.hysteresis_bars = hysteresis_bars

        self.exposure: Dict[str, float] = {
            "bull": exposure_bull,
            "neutral": exposure_neutral,
            "bear": exposure_bear,
        }

        # 滯後緩衝：記錄最近 N 次原始研判結果
        self._raw_signals: deque[str] = deque(maxlen=hysteresis_bars)
        self._current_regime: str = "neutral"

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def detect(
        self,
        index_prices: pd.Series,
        realized_vol: float = 0.0,
    ) -> Tuple[str, float]:
        """
        根據指數日線價格序列和已實現波動率研判市況。

        Parameters
        ----------
        index_prices : 指數日線收盤價序列（pd.Series，按時間升序）
                       例如 SPY / ^HSI / 恒生指數
        realized_vol : 年化已實現波動率（由 RealTimeMetrics 提供）

        Returns
        -------
        (regime_name, max_exposure)
          regime_name  : "bull" | "neutral" | "bear"
          max_exposure : 對應的最大組合倉位（0.05 ~ 0.40）
        """
        if index_prices is None or len(index_prices) < self.slow_period:
            # 數據不足時返回保守中性
            logger.debug(
                f"RegimeDetector: 指數數據不足（{len(index_prices) if index_prices is not None else 0} < {self.slow_period}），返回 neutral"
            )
            return "neutral", self.exposure["neutral"]

        raw = self._classify(index_prices, realized_vol)
        self._raw_signals.append(raw)

        # 滯後：只有 N 次連續相同研判才切換
        if len(self._raw_signals) >= self.hysteresis_bars:
            signals = list(self._raw_signals)
            if all(s == signals[0] for s in signals):
                if self._current_regime != signals[0]:
                    logger.info(
                        f"RegimeDetector: 市況切換  {self._current_regime} → {signals[0]}  "
                        f"(vol={realized_vol:.2%})"
                    )
                    self._current_regime = signals[0]

        max_exp = self.exposure[self._current_regime]
        logger.debug(
            f"RegimeDetector: regime={self._current_regime}  "
            f"max_exposure={max_exp:.0%}  vol={realized_vol:.2%}"
        )
        return self._current_regime, max_exp

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _classify(self, prices: pd.Series, vol: float) -> str:
        """原始研判（無滯後）：返回 bull / neutral / bear。"""
        fast_ma = float(prices.rolling(self.fast_period).mean().iloc[-1])
        slow_ma = float(prices.rolling(self.slow_period).mean().iloc[-1])
        last_price = float(prices.iloc[-1])

        trend_up = fast_ma > slow_ma
        low_vol = vol < self.vol_threshold
        high_vol = vol > self.vol_threshold * 1.5

        logger.debug(
            f"RegimeDetector raw: price={last_price:.2f}  "
            f"fast_ma={fast_ma:.2f}  slow_ma={slow_ma:.2f}  "
            f"trend_up={trend_up}  vol={vol:.2%}  low_vol={low_vol}  high_vol={high_vol}"
        )

        if trend_up and low_vol:
            return "bull"
        if not trend_up or high_vol:
            return "bear"
        return "neutral"

    @property
    def current_regime(self) -> str:
        """當前確定的市況狀態。"""
        return self._current_regime

    @property
    def current_exposure(self) -> float:
        """當前對應的最大組合倉位上限。"""
        return self.exposure[self._current_regime]

    def summary(self) -> Dict[str, Any]:
        return {
            "regime": self._current_regime,
            "max_exposure": self.exposure[self._current_regime],
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "vol_threshold": self.vol_threshold,
            "exposure_map": dict(self.exposure),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RegimeDetector":
        """從 config.yaml risk.smart_risk 塊構建實例。"""
        exp = cfg.get("exposure", {})
        return cls(
            fast_period=cfg.get("regime_fast_ma", 20),
            slow_period=cfg.get("regime_slow_ma", 60),
            vol_threshold=cfg.get("regime_vol_threshold", 0.25),
            hysteresis_bars=cfg.get("regime_hysteresis_bars", 3),
            exposure_bull=exp.get("bull", 0.40),
            exposure_neutral=exp.get("neutral", 0.20),
            exposure_bear=exp.get("bear", 0.05),
        )
