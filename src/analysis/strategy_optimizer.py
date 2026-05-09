"""
策略自動升級（僅輸出建議，不直接改代碼）

目標
----
每天/每週收盤後，基於最近表現自動產出：
- 微調建議：止損/止盈、最大倉位、再平衡頻率、因子權重等（以 config 建議形式給出）
- 淘汰建議：某些因子/子信號在最近窗口持續失效（勝率/信息比下降）
- 新規律提示：何種市況下策略更有效（趨勢/震盪/熊市），提出加入過濾器/分支策略的建議

設計約束
--------
- 不直接修改策略源碼（避免 AI 自動改壞策略）
- 輸出「可審計」的規則化建議（why + evidence + action）
- 能在數據不足時退化為保守建議
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TuningSuggestion:
    """單條調參/策略調整建議。"""

    category: str                # "risk" | "signal" | "execution" | "regime" | "data"
    title: str
    evidence: str                # 為什麼（證據/指標）
    action: str                  # 做什麼（具體可操作）
    severity: str = "medium"     # "low" | "medium" | "high"


@dataclass
class OptimizationResult:
    """一次自動優化結果。"""

    period: str
    headline: str
    suggestions: List[TuningSuggestion] = field(default_factory=list)
    config_patch: Dict[str, Any] = field(default_factory=dict)  # 建議寫回 config 的 patch（由人確認後再改）

    def to_text(self) -> str:
        lines = [self.headline, ""]
        if not self.suggestions:
            lines.append("無需調整：近期表現穩定，建議保持參數不變。")
            return "\n".join(lines)

        lines.append("── 建議清單 ──")
        for s in self.suggestions:
            lines.append(f"[{s.severity.upper():>6}] ({s.category}) {s.title}")
            lines.append(f"  證據：{s.evidence}")
            lines.append(f"  建議：{s.action}")
            lines.append("")

        if self.config_patch:
            lines.append("── 建議配置 Patch（人工確認後再落地）──")
            lines.append(str(self.config_patch))
        return "\n".join(lines).strip()


class StrategyOptimizer:
    """
    基於最近窗口的表現（勝率、均值回撤、歸因結果、黑天鵝觸發等）生成調參建議。
    """

    def __init__(
        self,
        *,
        min_trades: int = 5,
        low_win_rate: float = 0.45,
        high_drawdown: float = 0.08,
        high_exec_issue_ratio: float = 0.30,
    ) -> None:
        self.min_trades = min_trades
        self.low_win_rate = low_win_rate
        self.high_drawdown = high_drawdown
        self.high_exec_issue_ratio = high_exec_issue_ratio

    def propose(
        self,
        *,
        period: str,
        review_summary: Optional[Dict[str, Any]] = None,
        attribution_summary: Optional[Dict[str, Any]] = None,
        risk_dashboard: Optional[Dict[str, Any]] = None,
        current_config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        生成策略升級建議。

        review_summary:
          - total_trades, win_rate, signal_issues, exec_issues, market_issues, overall_verdict
        attribution_summary:
          - dominant_source, avg_trend, avg_revert, avg_alpha, total_return
        risk_dashboard:
          - volatility, max_drawdown, var_95, hhi, regime, max_exposure, is_halted
        """
        res = OptimizationResult(
            period=period,
            headline=f"【策略升級建議】{period}（僅建議，不自動改代碼）",
        )

        review = review_summary or {}
        attr = attribution_summary or {}
        dash = risk_dashboard or {}
        cfg = current_config or {}

        total_trades = int(review.get("total_trades", 0) or 0)
        win_rate = float(review.get("win_rate", 0.0) or 0.0)
        exec_issues = int(review.get("exec_issues", 0) or 0)
        signal_issues = int(review.get("signal_issues", 0) or 0)
        market_issues = int(review.get("market_issues", 0) or 0)

        max_dd = float(dash.get("max_drawdown", 0.0) or 0.0)
        vol = float(dash.get("volatility", 0.0) or 0.0)
        regime = str(dash.get("regime", "") or "")

        dominant = str(attr.get("dominant_source", "") or "")

        # ----------- 基礎健壯性：交易數太少不做激進調參 -----------
        if total_trades < self.min_trades:
            res.suggestions.append(
                TuningSuggestion(
                    category="data",
                    title="近期交易樣本偏少，避免過擬合調參",
                    evidence=f"最近窗口僅 {total_trades} 筆交易（< {self.min_trades}）",
                    action="保持策略參數不變；先累積更多樣本，再做因子淘汰/權重調整。",
                    severity="low",
                )
            )

        # ----------- 執行問題：滑點/成交偏差 -----------
        if total_trades >= 1:
            exec_ratio = exec_issues / max(total_trades, 1)
            if exec_ratio >= self.high_exec_issue_ratio:
                res.suggestions.append(
                    TuningSuggestion(
                        category="execution",
                        title="執行偏差偏多，優先優化下單方式",
                        evidence=f"執行問題 {exec_issues}/{total_trades}（{exec_ratio:.0%}）",
                        action="港股限價滑點可先從 0.30% 下調到 0.20%，並在成交不足時再回調；或增加分批下單避免一次性沖擊。",
                        severity="high",
                    )
                )
                # 建議 patch（不直接落地）
                fp = (cfg.get("execution", {}) or {}).get("futu_paper", {}) if cfg else {}
                current_slip = fp.get("slippage_bps", 30)
                if isinstance(current_slip, (int, float)) and current_slip >= 30:
                    res.config_patch.setdefault("execution", {}).setdefault("futu_paper", {})["slippage_bps"] = 20

        # ----------- 信號問題：勝率/錯誤方向 -----------
        if total_trades >= self.min_trades and win_rate < self.low_win_rate:
            res.suggestions.append(
                TuningSuggestion(
                    category="signal",
                    title="勝率偏低，信號可能進入衰退期",
                    evidence=f"勝率 {win_rate:.0%}（低於 {self.low_win_rate:.0%}）且信號問題 {signal_issues} 筆",
                    action="建議加入「市況過濾器」：在熊市/高波動時提高觸發門檻（例如要求多個子信號同時成立），或降低 top_n 避免弱信號。",
                    severity="high",
                )
            )

        # ----------- 行情問題：熊市下跌拖累 -----------
        if market_issues > 0 and regime.lower() == "bear":
            res.suggestions.append(
                TuningSuggestion(
                    category="regime",
                    title="熊市拖累明顯，建議進一步降低總敞口或增加空倉規則",
                    evidence=f"市況={regime}，行情問題 {market_issues} 筆，回撤={max_dd:.2%}，波動率={vol:.2%}",
                    action="可把 bear exposure 從 5% 下調到 0%（完全空倉），或啟用更嚴格的黑天鵝緊急倉位。",
                    severity="medium",
                )
            )
            res.config_patch.setdefault("risk", {}).setdefault("smart_risk", {}).setdefault("exposure", {})["bear"] = 0.0

        # ----------- 風險過高：回撤/波動 -----------
        if max_dd >= self.high_drawdown:
            res.suggestions.append(
                TuningSuggestion(
                    category="risk",
                    title="回撤偏高，建議收緊單票權重或止損",
                    evidence=f"滾動最大回撤 {max_dd:.2%}（≥ {self.high_drawdown:.0%}）",
                    action="先把 max_position_weight 下調 2-3 個百分點；止損可由 8% 收緊到 6%-7%（避免深回撤）。",
                    severity="high",
                )
            )
            current_mp = float((cfg.get("risk", {}) or {}).get("max_position_weight", 0.10) or 0.10) if cfg else 0.10
            res.config_patch.setdefault("risk", {})["max_position_weight"] = max(current_mp - 0.02, 0.05)

        # ----------- 歸因：震盪行情下信號失效 -----------
        if dominant and "震盪" in dominant:
            res.suggestions.append(
                TuningSuggestion(
                    category="signal",
                    title="收益主要受震盪影響，建議降低交易頻率/增加趨勢濾網",
                    evidence=f"歸因顯示主導來源：{dominant}",
                    action="增加趨勢濾網：僅在價格位於長期均線上方時允許做多；或把短線信號 trigger_mode 提升至 majority/all。",
                    severity="medium",
                )
            )

        # 收尾：如果沒有任何建議，給出保持策略的結論
        if not res.suggestions:
            res.headline += "：近期表現平穩，暫不建議改動。"

        return res

