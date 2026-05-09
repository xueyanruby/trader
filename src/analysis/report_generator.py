"""
日報/週報：結論先行 + 改進建議（你不用看數據）

此模塊把：
- 逐筆復盤（TradeReviewer）
- 收益歸因（ReturnAttributor）
- 風控儀表板（RiskManager.risk_dashboard / RealTimeMetrics）
- 策略升級建議（StrategyOptimizer）

整合成「一封能直接看結論」的報告正文。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


class DailyReportGenerator:
    """
    生成收盤後「自动复盘 + 归因 + 改进建议」日报。
    """

    def build_daily_review(
        self,
        *,
        date_str: str,
        portfolio_summary: Dict[str, Any],
        risk_dashboard: Dict[str, Any],
        review_summary: Optional[Dict[str, Any]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        optimizer_text: str = "",
    ) -> Dict[str, Any]:
        """
        返回結構化日報，供 MessageBuilder 轉成郵件正文。
        """
        review_summary = review_summary or {}
        attribution = attribution or {}

        conclusion = self._conclusion_lines(
            date_str=date_str,
            portfolio_summary=portfolio_summary,
            risk_dashboard=risk_dashboard,
            review_summary=review_summary,
            attribution=attribution,
        )

        improvements = self._improvement_lines(
            review_summary=review_summary,
            attribution=attribution,
            risk_dashboard=risk_dashboard,
        )

        return {
            "date": date_str,
            "conclusion": conclusion,
            "improvements": improvements,
            "portfolio_summary": portfolio_summary,
            "risk_dashboard": risk_dashboard,
            "trade_review": review_summary,
            "attribution": attribution,
            "optimizer": optimizer_text,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def build_weekly_upgrade(
        self,
        *,
        period: str,
        weekly_return_pct: float,
        weekly_attribution: Dict[str, Any],
        optimizer_text: str,
    ) -> Dict[str, Any]:
        return {
            "period": period,
            "weekly_return_pct": weekly_return_pct,
            "weekly_attribution": weekly_attribution,
            "optimizer": optimizer_text,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    # ------------------------------------------------------------------
    # 私有：結論與建議
    # ------------------------------------------------------------------

    @staticmethod
    def _conclusion_lines(
        *,
        date_str: str,
        portfolio_summary: Dict[str, Any],
        risk_dashboard: Dict[str, Any],
        review_summary: Dict[str, Any],
        attribution: Dict[str, Any],
    ) -> List[str]:
        equity = float(portfolio_summary.get("equity", 0) or 0)
        cash = float(portfolio_summary.get("cash", 0) or 0)
        regime = str(risk_dashboard.get("regime", "N/A") or "N/A")
        max_exp = float(risk_dashboard.get("max_exposure", 0) or 0)
        vol = float(risk_dashboard.get("volatility", 0) or 0)
        dd = float(risk_dashboard.get("max_drawdown", 0) or 0)
        var95 = float(risk_dashboard.get("var_95", 0) or 0)

        total_trades = int(review_summary.get("total_trades", 0) or 0)
        win_rate = float(review_summary.get("win_rate", 0.0) or 0.0)
        overall_verdict = str(review_summary.get("overall_verdict", "") or "")

        dominant_source = str(attribution.get("dominant_source", "") or "")
        verdict = str(attribution.get("verdict", "") or "")

        lines = [
            f"✅【今日结论】{date_str}",
            f"- 账户净值：${equity:,.2f}（现金 ${cash:,.2f}）",
            f"- 市况：{regime.upper()}｜最大仓位上限：{max_exp:.0%}",
            f"- 风险：波动率 {vol:.2%}｜回撤 {dd:.2%}｜VaR95 {var95:.2%}",
        ]
        if total_trades > 0:
            lines.append(f"- 交易质量：{total_trades} 笔｜胜率 {win_rate:.0%}")
        if dominant_source:
            lines.append(f"- 收益归因：主要来自「{dominant_source}」")
        if verdict:
            lines.append(f"- 归因解读：{verdict}")
        if overall_verdict:
            lines.append(f"- 复盘结论：{overall_verdict}")
        return lines

    @staticmethod
    def _improvement_lines(
        *,
        review_summary: Dict[str, Any],
        attribution: Dict[str, Any],
        risk_dashboard: Dict[str, Any],
    ) -> List[str]:
        suggestions: List[str] = []

        # 復盤優先
        top_lesson = str(review_summary.get("top_lesson", "") or "")
        if top_lesson:
            suggestions.append(f"- 交易改进：{top_lesson}")

        # 歸因
        dominant = str(attribution.get("dominant_source", "") or "")
        if dominant and "震盪" in dominant:
            suggestions.append("- 策略改进：震荡市信号更易失效，建议加趋势过滤/降低交易频率。")

        # 風控
        dd = float(risk_dashboard.get("max_drawdown", 0) or 0)
        if dd >= 0.08:
            suggestions.append("- 风控改进：回撤偏高，建议收紧单票上限或止损阈值，降低尾部风险。")

        if not suggestions:
            suggestions.append("- 改进建议：暂无明显问题，建议保持参数不变并持续观察。")

        return suggestions

