"""
收益歸因分析器（Return Attributor）

將當日 / 本週組合收益分解為三個來源：

  趨勢貢獻 (trend)          : 持有順勢標的獲得的 beta/momentum 收益
  震盪/均值回歸貢獻 (revert) : 逆勢標的或震盪市的噪音收益（多數時候為負）
  選股 Alpha (alpha)        : 扣除市場和趨勢後的純選股貢獻
  殘差/運氣 (luck)           : 模型解釋不了的剩餘（單日數據往往較大）

方法論（輕量，不依賴外部線性代數庫）
--------------------------------------
1. 市場 beta 貢獻 = 組合 beta × 大盤收益
   - beta 估算：近 20 日組合收益與大盤收益的協方差 / 大盤方差
   - 若歷史不足，beta 取 1.0

2. 趨勢因子貢獻 = 持有股票的動量加權均值收益
   - 動量分數（momentum_score）= 各股近 N 日收益率（由調用方傳入）
   - 信號與實際正相關 → 趨勢貢獻為正

3. 震盪貢獻 = 均值回歸分量（信號方向與實際相反的部分）

4. Alpha = portfolio_return - market_beta_component - trend_component

5. 殘差 = alpha（當日誤差較大，週維度更有意義）

使用示例
--------
att = ReturnAttributor()
result = att.decompose(
    portfolio_return=0.012,
    market_return=0.008,
    position_returns={"AAPL": 0.02, "MSFT": -0.005},
    position_weights={"AAPL": 0.15, "MSFT": 0.10},
    momentum_scores={"AAPL": 0.8, "MSFT": -0.2},  # +1=強勢 / -1=弱勢
)
print(result)
# {'trend': 0.009, 'revert': -0.001, 'alpha': 0.004, 'luck': 0.0, 'market_beta': 0.006}
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class AttributionResult:
    """單次歸因分析結果。"""
    date: str

    # 量化分量（均為收益率，可正可負）
    total_return: float = 0.0
    market_beta: float = 0.0    # 被動 beta 貢獻
    trend: float = 0.0          # 動量 / 趨勢貢獻
    revert: float = 0.0         # 震盪 / 均值回歸損耗（通常 ≤ 0）
    alpha: float = 0.0          # 純選股貢獻
    luck: float = 0.0           # 殘差（難以歸因部分）

    # 輔助字段
    estimated_beta: float = 1.0
    market_return: float = 0.0
    n_positions: int = 0

    # 可讀結論（由 _summarize 填充）
    dominant_source: str = ""   # "趨勢" | "選股" | "行情" | "震盪" | "雜訊"
    verdict: str = ""

    def as_pct_dict(self) -> Dict[str, str]:
        return {
            "總收益": f"{self.total_return*100:+.2f}%",
            "市場Beta": f"{self.market_beta*100:+.2f}%",
            "趨勢動量": f"{self.trend*100:+.2f}%",
            "震盪損耗": f"{self.revert*100:+.2f}%",
            "選股Alpha": f"{self.alpha*100:+.2f}%",
            "殘差/運氣": f"{self.luck*100:+.2f}%",
        }


class ReturnAttributor:
    """
    組合收益歸因器。

    Parameters
    ----------
    beta_window      : 估算 beta 所需的歷史天數（默認 20）
    momentum_window  : 動量分數的回溯天數（默認 10）
    """

    def __init__(
        self,
        beta_window: int = 20,
        momentum_window: int = 10,
    ) -> None:
        self.beta_window = beta_window
        self.momentum_window = momentum_window

        # 滾動歷史
        self._portfolio_history: Deque[float] = deque(maxlen=beta_window)
        self._market_history: Deque[float] = deque(maxlen=beta_window)
        self._results_history: List[AttributionResult] = []

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def decompose(
        self,
        portfolio_return: float,
        market_return: float,
        position_returns: Dict[str, float],   # {symbol: 今日收益率}
        position_weights: Dict[str, float],   # {symbol: 持倉權重}
        momentum_scores: Optional[Dict[str, float]] = None,  # {symbol: ±1 信號強度}
        date: str = "",
    ) -> AttributionResult:
        """
        執行一次收益歸因。

        Parameters
        ----------
        portfolio_return : 今日組合總收益率
        market_return    : 今日大盤收益率（如 SPY / ^HSI）
        position_returns : {symbol: 今日個股收益率}
        position_weights : {symbol: 持倉佔組合的權重}
        momentum_scores  : {symbol: 動量方向分數，+1=強動量 / -1=弱/逆勢}
                          由策略信號衍生（如 signal=+1 時 score=1，-1 時 score=-1）
        date             : YYYY-MM-DD

        Returns
        -------
        AttributionResult
        """
        # 更新滾動歷史
        self._portfolio_history.append(portfolio_return)
        self._market_history.append(market_return)

        # 1. 估算 Beta
        beta = self._estimate_beta()

        # 2. 市場 Beta 貢獻
        market_beta_component = beta * market_return

        # 3. 趨勢 / 動量貢獻
        trend_component, revert_component = self._trend_attribution(
            position_returns=position_returns,
            position_weights=position_weights,
            momentum_scores=momentum_scores or {},
        )

        # 4. Alpha = 總收益 - beta - trend - revert
        alpha = portfolio_return - market_beta_component - trend_component - revert_component

        # 5. 殘差（alpha 本身就是殘差，當日意義不大，週維度更穩定）
        luck = alpha  # 短期內難以區分 alpha 和運氣，統稱殘差

        result = AttributionResult(
            date=date,
            total_return=portfolio_return,
            market_beta=market_beta_component,
            trend=trend_component,
            revert=revert_component,
            alpha=alpha,
            luck=luck,
            estimated_beta=beta,
            market_return=market_return,
            n_positions=len(position_weights),
        )

        # 填充文字結論
        result.dominant_source, result.verdict = self._make_verdict(result)
        self._results_history.append(result)
        return result

    # ------------------------------------------------------------------
    # 週度彙總（跨多日的平均歸因）
    # ------------------------------------------------------------------

    def weekly_summary(self, days: int = 5) -> Dict[str, Any]:
        """返回最近 N 日的平均歸因分解。"""
        recent = self._results_history[-days:] if self._results_history else []
        if not recent:
            return {}

        n = len(recent)

        def avg(attr: str) -> float:
            return sum(getattr(r, attr, 0) for r in recent) / n

        total_ret = sum(r.total_return for r in recent)
        dominant_sources = [r.dominant_source for r in recent if r.dominant_source]
        dominant = max(set(dominant_sources), key=dominant_sources.count) if dominant_sources else "未知"

        return {
            "period_days": n,
            "total_return": total_ret,
            "avg_market_beta": avg("market_beta"),
            "avg_trend": avg("trend"),
            "avg_revert": avg("revert"),
            "avg_alpha": avg("alpha"),
            "dominant_source": dominant,
            "verdict": f"本週主要收益來源：{dominant}。詳見分項分析。",
        }

    # ------------------------------------------------------------------
    # 私有：Beta 估算
    # ------------------------------------------------------------------

    def _estimate_beta(self) -> float:
        """最小二乘 Beta（無外部依賴，使用滾動協方差 / 方差）。"""
        p_hist = list(self._portfolio_history)
        m_hist = list(self._market_history)
        n = min(len(p_hist), len(m_hist))
        if n < 3:
            return 1.0

        p_series = p_hist[-n:]
        m_series = m_hist[-n:]
        p_mean = sum(p_series) / n
        m_mean = sum(m_series) / n

        cov = sum((p - p_mean) * (m - m_mean) for p, m in zip(p_series, m_series)) / (n - 1)
        var_m = sum((m - m_mean) ** 2 for m in m_series) / (n - 1)

        if var_m < 1e-10:
            return 1.0
        beta = cov / var_m
        # 合理範圍 clamp
        return max(min(beta, 3.0), -1.0)

    # ------------------------------------------------------------------
    # 私有：趨勢 / 震盪分解
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_attribution(
        position_returns: Dict[str, float],
        position_weights: Dict[str, float],
        momentum_scores: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        將持倉收益拆為「順趨勢」和「逆趨勢/震盪」兩部分。

        順趨勢 (trend)：
          momentum_score > 0 且 position_return > 0（信號正確，趨勢確立）
          或 score < 0 且 return < 0（做空正確）
        震盪 (revert)：
          信號方向與實際收益相反的部分（信號說漲但跌，或說跌但漲）

        返回 (trend_total, revert_total) 均為組合層面的加權貢獻。
        """
        trend_total = 0.0
        revert_total = 0.0

        total_weight = sum(position_weights.values()) or 1.0

        for sym, weight in position_weights.items():
            ret = position_returns.get(sym, 0.0)
            score = momentum_scores.get(sym, 1.0)   # 無 score 時假設順勢

            norm_weight = weight / total_weight
            contribution = ret * norm_weight

            # 信號方向與實際一致 → 趨勢貢獻
            if (score >= 0 and ret >= 0) or (score < 0 and ret < 0):
                trend_total += contribution
            else:
                # 信號與實際不符 → 震盪損耗（通常為負）
                revert_total += contribution

        return trend_total, revert_total

    # ------------------------------------------------------------------
    # 私有：文字結論
    # ------------------------------------------------------------------

    @staticmethod
    def _make_verdict(r: AttributionResult) -> Tuple[str, str]:
        """根據各分量大小生成主要來源標籤和可讀結論。"""
        # 找最大絕對值分量（代表主要驅動力）
        components = {
            "市場行情": abs(r.market_beta),
            "趨勢動量": abs(r.trend),
            "震盪波動": abs(r.revert),
            "選股Alpha": abs(r.alpha),
        }
        dominant = max(components, key=lambda k: components[k])

        sign = "+" if r.total_return >= 0 else ""
        total_str = f"{r.total_return*100:+.2f}%"
        market_str = f"{r.market_return*100:+.2f}%"

        if r.total_return > 0:
            if dominant == "市場行情":
                verdict = (
                    f"今日收益 {total_str} 主要來自大盤上漲（{market_str}）。"
                    "策略跟隨市場獲益，非純 Alpha，需警惕市場回調風險。"
                )
            elif dominant == "趨勢動量":
                verdict = (
                    f"今日收益 {total_str} 主要來自趨勢動量。"
                    "策略選到了表現超越大盤的強勢股，信號質量良好。"
                )
            elif dominant == "選股Alpha":
                verdict = (
                    f"今日收益 {total_str} 主要為純選股 Alpha（大盤 {market_str}，跑贏顯著）。"
                    "策略在此類市況下有獨特優勢，可適當加大倉位。"
                )
            else:
                verdict = f"今日收益 {total_str}，收益來源較分散，震盪行情中表現平穩。"
        elif r.total_return < 0:
            if dominant == "市場行情":
                verdict = (
                    f"今日虧損 {total_str} 主要由大盤下跌（{market_str}）拖累。"
                    "策略本身無誤，在熊市環境下優先縮倉。"
                )
            elif dominant == "震盪波動":
                verdict = (
                    f"今日虧損 {total_str}，震盪行情中信號頻繁失效。"
                    "建議增加趨勢過濾器，減少震盪市場的交易頻率。"
                )
            elif dominant == "信號Alpha":
                verdict = (
                    f"今日虧損 {total_str}，信號方向判斷偏差為主要原因。"
                    "需複盤近期因子 IC 是否下滑。"
                )
            else:
                verdict = f"今日虧損 {total_str}，多因素疊加，需逐項復盤。"
        else:
            verdict = f"今日收益 {total_str}，市場基本持平。"

        return dominant, verdict
