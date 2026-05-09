"""
黑天鵝事件探測器（Black Swan Detector）

識別歷史中未曾出現過的極端波動，滿足任意一條即觸發：

  1. Z-score 極端       : 當日收益率超出歷史分佈 ±N 個標準差（默認 3.5σ）
  2. 波動率驟升         : 短期已實現波動率 ≥ 長期均值的 M 倍（默認 3.0×）
  3. VaR 突破           : 當日損失超過歷史 99% VaR 邊界

觸發後建議行動：
  - 立即向 notifier 推送 send_black_swan_alert
  - 調用 broker.rebalance({}) 平倉，或按 emergency_exposure 縮倉

同一日只觸發一次（內置日期去重）。

使用方式
--------
detector = BlackSwanDetector(zscore_threshold=3.5)
triggered, reason = detector.check(
    today_return=-0.12,
    return_history=historical_returns_series,
    current_vol=0.80,
    long_term_vol=0.20,
)
if triggered:
    notifier.send_black_swan_alert(today_return, reason, action="emergency_reduce")
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from loguru import logger


class BlackSwanDetector:
    """
    三重觸發機制的黑天鵝探測器。

    Parameters
    ----------
    zscore_threshold   : Z-score 閾值（絕對值），默認 3.5σ
    vol_spike_factor   : 當前波動率 / 長期均值的倍數觸發閾值，默認 3.0
    var_confidence     : VaR 突破檢測的置信度，默認 0.99（即超 99% VaR 才觸發）
    min_history        : 計算 Z-score / VaR 所需的最少歷史觀測數，默認 20
    """

    def __init__(
        self,
        zscore_threshold: float = 3.5,
        vol_spike_factor: float = 3.0,
        var_confidence: float = 0.99,
        min_history: int = 20,
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.vol_spike_factor = vol_spike_factor
        self.var_confidence = var_confidence
        self.min_history = min_history

        # 同日去重
        self._last_triggered_date: Optional[date] = None

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def check(
        self,
        today_return: float,
        return_history: "pd.Series | list[float]",
        current_vol: float = 0.0,
        long_term_vol: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        執行三重黑天鵝檢測。

        Parameters
        ----------
        today_return    : 今日組合收益率（如 -0.12 表示跌 12%）
        return_history  : 歷史日收益率序列（用於計算 Z-score / VaR）
        current_vol     : 當前短期年化波動率（由 RealTimeMetrics 提供）
        long_term_vol   : 長期平均年化波動率（用於波動率驟升比較）

        Returns
        -------
        (triggered: bool, reason: str)
        """
        today = date.today()

        # 同日只觸發一次
        if self._last_triggered_date == today:
            return False, ""

        # 轉換歷史序列
        if isinstance(return_history, pd.Series):
            hist = return_history.dropna().tolist()
        else:
            hist = [r for r in return_history if r is not None]

        triggered, reason = self._run_checks(today_return, hist, current_vol, long_term_vol)

        if triggered:
            self._last_triggered_date = today
            logger.warning(
                f"[BlackSwanDetector] 🦢 黑天鵝觸發！"
                f"  today_return={today_return:.2%}"
                f"  current_vol={current_vol:.2%}"
                f"  long_term_vol={long_term_vol:.2%}"
                f"  原因={reason}"
            )

        return triggered, reason

    # ------------------------------------------------------------------
    # 私有：三重檢測
    # ------------------------------------------------------------------

    def _run_checks(
        self,
        today_return: float,
        hist: list,
        current_vol: float,
        long_term_vol: float,
    ) -> Tuple[bool, str]:

        reasons = []

        # ── 檢測 1：Z-score 極端 ──────────────────────────────────────
        if len(hist) >= self.min_history:
            z = self._zscore(today_return, hist)
            if abs(z) >= self.zscore_threshold:
                reasons.append(
                    f"Z-score 極端（z={z:.2f}，閾值±{self.zscore_threshold}σ，"
                    f"日收益={today_return:.2%}）"
                )
                logger.debug(f"BlackSwan check1 TRIGGERED: z={z:.2f}")
            else:
                logger.debug(f"BlackSwan check1 ok: z={z:.2f}")
        else:
            logger.debug(f"BlackSwan check1 skip: 歷史觀測不足 ({len(hist)} < {self.min_history})")

        # ── 檢測 2：波動率驟升 ────────────────────────────────────────
        if current_vol > 0 and long_term_vol > 0:
            ratio = current_vol / long_term_vol
            if ratio >= self.vol_spike_factor:
                reasons.append(
                    f"波動率驟升（當前 vol={current_vol:.2%}，"
                    f"長期均值={long_term_vol:.2%}，"
                    f"倍數={ratio:.1f}×，閾值={self.vol_spike_factor}×）"
                )
                logger.debug(f"BlackSwan check2 TRIGGERED: vol_ratio={ratio:.2f}")
            else:
                logger.debug(f"BlackSwan check2 ok: vol_ratio={ratio:.2f}")
        else:
            logger.debug("BlackSwan check2 skip: 波動率數據不足")

        # ── 檢測 3：VaR 突破 ──────────────────────────────────────────
        if len(hist) >= self.min_history:
            var = self._historical_var(hist, self.var_confidence)
            if today_return < -var:
                reasons.append(
                    f"VaR 突破（日收益={today_return:.2%}，"
                    f"VaR{int(self.var_confidence*100)}%=-{var:.2%}）"
                )
                logger.debug(f"BlackSwan check3 TRIGGERED: today={today_return:.2%} < -VaR={-var:.2%}")
            else:
                logger.debug(f"BlackSwan check3 ok: today={today_return:.2%}, VaR={var:.2%}")

        if reasons:
            return True, " | ".join(reasons)
        return False, ""

    @staticmethod
    def _zscore(value: float, history: list) -> float:
        n = len(history)
        if n < 2:
            return 0.0
        mean = sum(history) / n
        variance = sum((r - mean) ** 2 for r in history) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return (value - mean) / std

    @staticmethod
    def _historical_var(history: list, confidence: float) -> float:
        """VaR（正數，表示損失幅度）。"""
        sorted_returns = sorted(history)
        idx = int((1 - confidence) * len(sorted_returns))
        return max(-sorted_returns[idx], 0.0)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @property
    def last_triggered_date(self) -> Optional[date]:
        return self._last_triggered_date

    def reset_daily(self) -> None:
        """手動重置日期去重（用於測試）。"""
        self._last_triggered_date = None

    def summary(self) -> Dict[str, Any]:
        return {
            "zscore_threshold": self.zscore_threshold,
            "vol_spike_factor": self.vol_spike_factor,
            "var_confidence": self.var_confidence,
            "last_triggered_date": str(self._last_triggered_date),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BlackSwanDetector":
        """從 config.yaml risk.smart_risk 塊構建實例。"""
        return cls(
            zscore_threshold=cfg.get("black_swan_zscore", 3.5),
            vol_spike_factor=cfg.get("black_swan_vol_spike", 3.0),
            var_confidence=cfg.get("black_swan_var_confidence", 0.99),
            min_history=cfg.get("black_swan_min_history", 20),
        )
