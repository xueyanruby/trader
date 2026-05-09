"""
實時風險指標計算器

持續接收 equity 快照，計算以下核心指標：
  - realized_volatility : 年化已實現波動率（20 日滾動）
  - rolling_max_drawdown : 滾動最大回撤
  - historical_var       : 歷史模擬法 VaR（默認 95% 置信度）
  - position_hhi         : Herfindahl-Hirschman 集中度指數（0-1）

使用方式
--------
rt = RealTimeMetrics(lookback_days=20)
for equity in equity_stream:
    metrics = rt.update(equity, weights={"AAPL": 0.3, "MSFT": 0.2})
    print(metrics["volatility"], metrics["max_drawdown"])
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Optional

from loguru import logger


class RealTimeMetrics:
    """
    持續計算投資組合實時風險指標。

    Parameters
    ----------
    lookback_days  : 滾動窗口（計算波動率 / 回撤 / VaR 的天數）
    var_confidence : VaR 置信度（默認 0.95）
    annualize_factor : 年化因子（默認 252 個交易日）
    """

    def __init__(
        self,
        lookback_days: int = 20,
        var_confidence: float = 0.95,
        annualize_factor: int = 252,
    ) -> None:
        self.lookback = lookback_days
        self.var_confidence = var_confidence
        self.annualize_factor = annualize_factor

        # 滾動窗口：equity 值序列（多一個，用於計算日收益率）
        self._equity_history: deque[float] = deque(maxlen=lookback_days + 1)
        # 日收益率序列（長度 = lookback_days）
        self._return_history: deque[float] = deque(maxlen=lookback_days)
        # 用於計算滾動峰值（max drawdown）
        self._peak_equity: float = 0.0

        # 快取最後計算的指標
        self._last_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 主更新接口
    # ------------------------------------------------------------------

    def update(
        self,
        equity: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        接收最新資產淨值，更新並返回所有指標快照。

        Parameters
        ----------
        equity  : 當前資產淨值（含持倉市值 + 現金）
        weights : 各持倉的最新目標權重 dict，用於計算 HHI 集中度

        Returns
        -------
        dict with keys: volatility, max_drawdown, var_95, hhi
        """
        if equity <= 0:
            return self._last_metrics

        # 更新峰值
        if equity > self._peak_equity:
            self._peak_equity = equity

        today_return: float = 0.0
        # 計算日收益率
        if self._equity_history:
            prev = self._equity_history[-1]
            if prev > 0:
                daily_return = (equity - prev) / prev
                today_return = daily_return
                self._return_history.append(daily_return)

        self._equity_history.append(equity)

        metrics: Dict[str, Any] = {
            "volatility": self.realized_volatility(),
            "max_drawdown": self.rolling_max_drawdown(equity),
            "var_95": self.historical_var(self.var_confidence),
            "hhi": self.position_hhi(weights or {}),
            "equity": equity,
            "today_return": today_return,
            "num_observations": len(self._return_history),
        }
        self._last_metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # 個別指標
    # ------------------------------------------------------------------

    def realized_volatility(self) -> float:
        """
        年化已實現波動率（標準差 × sqrt(annualize_factor)）。
        需要至少 2 個收益率觀測值，否則返回 0。
        """
        returns = list(self._return_history)
        n = len(returns)
        if n < 2:
            return 0.0
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        daily_std = math.sqrt(variance)
        return daily_std * math.sqrt(self.annualize_factor)

    def rolling_max_drawdown(self, current_equity: Optional[float] = None) -> float:
        """
        從歷史 equity 序列中計算最大回撤（峰谷比）。
        返回值為正數（例如 0.15 表示 15% 回撤）。
        """
        equities = list(self._equity_history)
        if current_equity is not None and (not equities or equities[-1] != current_equity):
            equities.append(current_equity)

        if len(equities) < 2:
            return 0.0

        max_drawdown = 0.0
        peak = equities[0]
        for eq in equities[1:]:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak
                if dd > max_drawdown:
                    max_drawdown = dd
        return max_drawdown

    def historical_var(self, confidence: float = 0.95) -> float:
        """
        歷史模擬法 VaR：在給定置信度下，單日最大損失的估計值。
        返回值為正數（例如 0.03 表示 3% 的 VaR）。
        需要至少 10 個觀測值，否則返回 0。
        """
        returns = list(self._return_history)
        if len(returns) < 10:
            return 0.0
        sorted_returns = sorted(returns)
        # 在 95% 置信度下，VaR 是最差 5% 的邊界值
        idx = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[idx]  # 轉為正數
        return max(var, 0.0)

    def position_hhi(self, weights: Dict[str, float]) -> float:
        """
        Herfindahl-Hirschman 集中度指數（HHI）。
        HHI = sum(w²)，範圍 [1/n, 1]，越接近 1 越集中。
        空倉時返回 0。
        """
        if not weights:
            return 0.0
        total = sum(weights.values())
        if total <= 0:
            return 0.0
        normalized = [w / total for w in weights.values()]
        return sum(w ** 2 for w in normalized)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def get_last(self) -> Dict[str, Any]:
        """返回最後一次 update() 的指標快照（無新數據時）。"""
        return self._last_metrics

    def reset(self) -> None:
        """重置所有歷史數據。"""
        self._equity_history.clear()
        self._return_history.clear()
        self._peak_equity = 0.0
        self._last_metrics = {}
        logger.debug("RealTimeMetrics: 已重置")

    def summary_str(self) -> str:
        """返回可讀的摘要字符串，用於日誌。"""
        m = self._last_metrics
        if not m:
            return "RealTimeMetrics: 暫無數據"
        return (
            f"波動率={m.get('volatility', 0):.2%}  "
            f"最大回撤={m.get('max_drawdown', 0):.2%}  "
            f"VaR95={m.get('var_95', 0):.2%}  "
            f"HHI={m.get('hhi', 0):.3f}"
        )
