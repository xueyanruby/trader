"""
通知模塊：郵件（SMTP）+ 企業微信 Webhook

支持的通知類型：
- 每日信號播報：策略生成的買入/賣出列表
- 倉位變動通知：再平衡後的訂單明細
- 止損警報：持倉觸碰止損線，需要立即操作
- 熔斷警報：組合回撤超過閾值，所有新單暫停
- 每日收盤報告：持倉市值、盈虧、風控指標摘要

使用方法
--------
from src.notifier import Notifier

notifier = Notifier(config["notify"])

# 發送止損警報
notifier.send_stop_loss_alert("AAPL", entry=150.0, current=136.5, loss_pct=-0.09)

# 發送每日信號
notifier.send_daily_signals(buy=["AAPL","MSFT"], sell=["TSLA"], hold=["NVDA"])

# 發送每日持倉報告
notifier.send_portfolio_report(summary)
"""

from __future__ import annotations

import json
import smtplib
import textwrap
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


# ---------------------------------------------------------------------------
# 消息內容構造器
# ---------------------------------------------------------------------------

class MessageBuilder:
    """將結構化數據組裝成可讀的文本消息。"""

    @staticmethod
    def daily_signals(
        strategy: str,
        date: str,
        buy: List[str],
        sell: List[str],
        hold: List[str],
    ) -> tuple[str, str]:
        """
        返回 (subject, body) 兩元組。
        subject 用於郵件主題，body 用於正文和微信消息。
        """
        subject = f"【交易信號】{date} {strategy}"

        lines = [
            f"策略：{strategy}",
            f"日期：{date}",
            "",
        ]
        if buy:
            lines.append(f"📈 買入（{len(buy)} 只）：{', '.join(buy)}")
        if sell:
            lines.append(f"📉 賣出（{len(sell)} 只）：{', '.join(sell)}")
        if hold:
            lines.append(f"⏸ 繼續持有（{len(hold)} 只）：{', '.join(hold)}")
        if not buy and not sell:
            lines.append("✅ 無需操作，組合無變動")

        lines += ["", "— Auto Trader"]
        body = "\n".join(lines)
        return subject, body

    @staticmethod
    def stop_loss_alert(
        symbol: str,
        entry_price: float,
        current_price: float,
        loss_pct: float,
        threshold_pct: float,
    ) -> tuple[str, str]:
        subject = f"【止損警報】{symbol} 已觸發止損線"
        body = textwrap.dedent(f"""
            ⚠️  止損警報

            股票代碼  ：{symbol}
            建倉價格  ：${entry_price:.2f}
            當前價格  ：${current_price:.2f}
            虧損幅度  ：{loss_pct*100:.2f}%
            止損閾值  ：{threshold_pct*100:.2f}%

            請立即評估是否平倉。

            — Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """).strip()
        return subject, body

    @staticmethod
    def circuit_breaker_alert(
        drawdown_pct: float,
        halt_threshold_pct: float,
        equity: float,
    ) -> tuple[str, str]:
        subject = "【熔斷警報】組合回撤超限，新單已暫停"
        body = textwrap.dedent(f"""
            🔴  組合熔斷警報

            當前回撤   ：{drawdown_pct*100:.2f}%
            熔斷閾值   ：{halt_threshold_pct*100:.2f}%
            當前資產淨值：${equity:,.2f}

            系統已自動暫停所有新訂單。
            如需恢復交易，請手動調用 risk_manager.reset_halt()。

            — Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """).strip()
        return subject, body

    @staticmethod
    def buy_signal_alert(
        symbol: str,
        price: float,
        signal_type: str,
        short_detail: Dict[str, Any],
        long_detail: Dict[str, Any],
    ) -> tuple[str, str]:
        """
        買入信號通知。
        signal_type: "short" | "long" | "both"
        """
        label_map = {
            "short": "📈 短線",
            "long": "🚀 長線",
            "both": "📈🚀 短線+長線",
        }
        label = label_map.get(signal_type, signal_type)
        subject = f"【買入信號】{symbol}  {label} 信號觸發"

        lines = [
            f"📌 股票代碼 ：{symbol}",
            f"💰 當前價格 ：{price:.4f}",
            f"📊 信號類型 ：{label}",
            "",
        ]

        if short_detail and signal_type in ("short", "both"):
            lines.append("── 短線信號詳情 ──")
            for k, v in short_detail.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        if long_detail and signal_type in ("long", "both"):
            lines.append("── 長線信號詳情 ──")
            for k, v in long_detail.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        lines += [
            "⚠️  以上為技術信號，不構成投資建議，請結合基本面自行判斷。",
            "",
            f"— Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        body = "\n".join(lines)
        return subject, body

    @staticmethod
    def buy_signal_digest(
        scan_count: int,
        rows: List[Dict[str, Any]],
        detail_limit: int = 8,
    ) -> tuple[str, str]:
        """
        全池技術掃描匯總郵件（多只股票一封，避免刷屏）。

        rows: 每项含 symbol, price, signal_type, short_detail, long_detail, short_fired, long_fired
        """
        n = len(rows)
        subject = f"【買入信號匯總】全池掃描 {scan_count} 只 → 本輪觸發 {n} 只"
        lines = [
            f"掃描標的數：{scan_count}",
            f"本輪觸發買入技術信號：{n} 只",
            "",
            "── 觸發列表 ──",
        ]
        for i, row in enumerate(rows):
            sym = row.get("symbol", "")
            price = row.get("price", 0.0)
            st = row.get("signal_type", "")
            lines.append(f"  {i + 1}. {sym}  價≈{float(price):.4f}  ({st})")

        lines.append("")
        lines.append("── 詳情（前若干只）──")
        for i, row in enumerate(rows[: max(0, detail_limit)]):
            sym = row.get("symbol", "")
            st = row.get("signal_type", "")
            lines.append(f"### {sym} ({st})")
            if row.get("short_fired") and row.get("short_detail"):
                lines.append("  短線:")
                for k, v in (row.get("short_detail") or {}).items():
                    lines.append(f"    {k}: {v}")
            if row.get("long_fired") and row.get("long_detail"):
                lines.append("  長線:")
                for k, v in (row.get("long_detail") or {}).items():
                    lines.append(f"    {k}: {v}")
            lines.append("")

        if n > detail_limit:
            lines.append(f"… 其餘 {n - detail_limit} 只僅列於上表，詳情請查日誌。")
            lines.append("")

        lines += [
            "⚠️  以上為技術信號，不構成投資建議。",
            f"— Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        body = "\n".join(lines)
        return subject, body

    @staticmethod
    def portfolio_report(summary: Dict[str, Any]) -> tuple[str, str]:
        date_str = datetime.now().strftime("%Y-%m-%d")
        subject = f"【每日報告】{date_str} 持倉摘要"

        equity = summary.get("equity", 0)
        cash = summary.get("cash", 0)
        num_pos = summary.get("num_positions", 0)
        positions = summary.get("positions", {})

        lines = [
            f"📊  {date_str} 收盤持倉報告",
            "",
            f"資產淨值   ：${equity:,.2f}",
            f"可用現金   ：${cash:,.2f}",
            f"持倉股票數  ：{num_pos}",
            "",
        ]

        if positions:
            lines.append("── 持倉明細 ──")
            for sym, pos in positions.items():
                pnl_pct = pos.get("unrealized_pnl_pct", "N/A")
                mv = pos.get("market_value", 0)
                lines.append(
                    f"  {sym:<10}  市值=${mv:>10,.2f}  浮盈={pnl_pct}"
                )

        lines += ["", "— Auto Trader"]
        body = "\n".join(lines)
        return subject, body

    @staticmethod
    def weekly_report(summary: Dict[str, Any]) -> tuple[str, str]:
        """
        週報消息。

        summary 必填字段：
          period        : "2025-05-05 ~ 2025-05-09"
          start_equity  : float
          end_equity    : float
          return_pct    : float  （可負）
          total_trades  : int
          fills         : List[dict]  （每條：symbol / side / qty / price）
          top_buys      : List[str]
          top_sells     : List[str]
        """
        period = summary.get("period", "本週")
        start_eq = summary.get("start_equity", 0)
        end_eq = summary.get("end_equity", 0)
        ret_pct = summary.get("return_pct", 0)
        total_trades = summary.get("total_trades", 0)
        fills = summary.get("fills", [])
        top_buys = summary.get("top_buys", [])
        top_sells = summary.get("top_sells", [])

        sign = "+" if ret_pct >= 0 else ""
        subject = f"【週報】{period}  模擬盤週度表現  {sign}{ret_pct*100:.2f}%"

        lines = [
            f"📅  週報：{period}",
            "",
            f"期初資產  ：${start_eq:,.2f}",
            f"期末資產  ：${end_eq:,.2f}",
            f"週收益率  ：{sign}{ret_pct*100:.2f}%",
            f"週交易筆數：{total_trades}",
            "",
        ]

        if top_buys:
            lines.append(f"📈 本週買入：{', '.join(top_buys)}")
        if top_sells:
            lines.append(f"📉 本週賣出：{', '.join(top_sells)}")

        if fills:
            lines.append("")
            lines.append("── 成交明細（最近 10 筆）──")
            for f in fills[-10:]:
                side_icon = "↑" if str(f.get("side", "")).lower() == "buy" else "↓"
                lines.append(
                    f"  {side_icon} {f.get('symbol', '?'):<10}  "
                    f"{str(f.get('side', '')).upper():4s}  "
                    f"qty={int(f.get('qty', 0)):6d}  "
                    f"price={f.get('price', 0):.3f}"
                )

        lines += [
            "",
            "⚠️  以上為模擬盤數據，不構成投資建議。",
            "",
            f"— Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        return subject, "\n".join(lines)

    @staticmethod
    def risk_dashboard_alert(dashboard: Dict[str, Any]) -> tuple[str, str]:
        """風控儀表板摘要——附在每日報告末尾。"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        regime = dashboard.get("regime", "N/A")
        regime_icons = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}
        icon = regime_icons.get(regime, "⬜")

        subject = f"【風控日報】{date_str} 市況={icon}{regime.upper()}"

        lines = [
            f"📊  {date_str} 風控儀表板",
            "",
            f"市況狀態   ：{icon} {regime.upper()}",
            f"最大倉位   ：{dashboard.get('max_exposure', 0):.0%}",
            f"年化波動率  ：{dashboard.get('volatility', 0):.2%}",
            f"滾動最大回撤：{dashboard.get('max_drawdown', 0):.2%}",
            f"VaR(95%)   ：{dashboard.get('var_95', 0):.2%}",
            f"集中度(HHI)：{dashboard.get('hhi', 0):.3f}",
            f"當前資產淨值：${dashboard.get('current_equity', 0):,.2f}",
            f"峰值資產淨值：${dashboard.get('peak_equity', 0):,.2f}",
            f"熔斷狀態   ：{'🚫 已熔斷' if dashboard.get('is_halted') else '✅ 正常'}",
            "",
            f"風控閾值   ：止損={dashboard.get('stop_loss_pct', 0):.0%}  "
            f"止盈={dashboard.get('take_profit_pct', 0):.0%}  "
            f"熔斷={dashboard.get('max_drawdown_halt', 0):.0%}",
            "",
            "— Auto Trader",
        ]
        return subject, "\n".join(lines)

    @staticmethod
    def black_swan_alert(
        return_pct: float,
        reason: str,
        action: str,
        equity: float = 0.0,
    ) -> tuple[str, str]:
        """黑天鵝事件緊急通知。"""
        subject = f"🦢【黑天鵝警報】組合觸發極端事件  {return_pct*100:.2f}%"
        lines = [
            "🦢  ⚠️  黑天鵝事件警報  ⚠️",
            "",
            f"觸發日期   ：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"當日收益率  ：{return_pct*100:.2f}%",
            f"觸發原因   ：{reason}",
            "",
            f"系統動作   ：{action}",
            f"當前資產淨值：${equity:,.2f}",
            "",
            "⚠️  以上為模擬盤自動風控操作，請核查倉位並評估市場狀況。",
            "",
            f"— Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        return subject, "\n".join(lines)

    @staticmethod
    def trade_executed(orders: List[Dict[str, Any]], regime: str = "") -> tuple[str, str]:
        """策略信號觸發後的下單明細通知。"""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        subject = f"【下單通知】{date_str}  共 {len(orders)} 筆模擬盤訂單"

        regime_txt = f"  市況：{regime.upper()}" if regime else ""
        lines = [
            f"🤖  策略信號執行完成{regime_txt}",
            f"時間：{date_str}",
            "",
            "── 訂單明細 ──",
        ]
        for o in orders:
            side_icon = "↑ 買入" if str(o.get("side", "")).lower() == "buy" else "↓ 賣出"
            status = o.get("status", "?")
            lines.append(
                f"  {side_icon}  {o.get('symbol', '?'):<10}  "
                f"qty={int(o.get('qty', 0)):6d}  "
                f"status={status}"
            )
        lines += ["", "（以上均為模擬盤，TrdEnv.SIMULATE）", "", "— Auto Trader"]
        return subject, "\n".join(lines)

    @staticmethod
    def daily_review(review: Dict[str, Any]) -> tuple[str, str]:
        """
        收盤后自动复盘日报（结论先行）。

        review 由 DailyReportGenerator.build_daily_review() 返回。
        """
        date_str = str(review.get("date") or datetime.now().strftime("%Y-%m-%d"))
        subject = f"【收盘复盘】{date_str} 结论 & 改进建议"

        conclusion = review.get("conclusion") or []
        improvements = review.get("improvements") or []
        optimizer = str(review.get("optimizer") or "").strip()

        lines: List[str] = []
        lines.extend(conclusion if isinstance(conclusion, list) else [str(conclusion)])
        lines.append("")
        lines.append("🛠【改进建议】")
        lines.extend(improvements if isinstance(improvements, list) else [str(improvements)])

        if optimizer:
            lines.append("")
            lines.append("🧠【策略升级建议（仅建议，不自动改代码）】")
            lines.append(optimizer)

        lines += ["", f"— Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
        return subject, "\n".join(lines)

    @staticmethod
    def weekly_strategy_upgrade(payload: Dict[str, Any]) -> tuple[str, str]:
        """
        周度策略升级总结（结论 + 下一步建议）。

        payload 由 DailyReportGenerator.build_weekly_upgrade() 返回。
        """
        period = str(payload.get("period") or "本週")
        weekly_ret = float(payload.get("weekly_return_pct", 0) or 0)
        sign = "+" if weekly_ret >= 0 else ""
        subject = f"【策略升级周报】{period}  {sign}{weekly_ret*100:.2f}%"

        lines = [
            f"📅 周期：{period}",
            f"周收益率：{sign}{weekly_ret*100:.2f}%",
            "",
        ]

        att = payload.get("weekly_attribution") or {}
        if isinstance(att, dict) and att:
            lines.append("📌 本周归因摘要：")
            dom = att.get("dominant_source", "")
            if dom:
                lines.append(f"- 主要来源：{dom}")
            if att.get("verdict"):
                lines.append(f"- 解读：{att.get('verdict')}")
            if att.get("total_return") is not None:
                try:
                    lines.append(f"- 总收益（5日合计）：{float(att.get('total_return'))*100:+.2f}%")
                except Exception:
                    pass
            lines.append("")

        optimizer = str(payload.get("optimizer") or "").strip()
        if optimizer:
            lines.append("🧠 策略升级建议：")
            lines.append(optimizer)

        lines += ["", f"— Auto Trader {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
        return subject, "\n".join(lines)


# ---------------------------------------------------------------------------
# 企業微信 Webhook 通知
# ---------------------------------------------------------------------------

class WeComNotifier:
    """
    企業微信群機器人 Webhook 通知。

    在企業微信群中添加機器人，複製 Webhook URL 填入配置：
    config.yaml → notify.wecom.webhook_url
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, content: str) -> bool:
        """
        發送 Markdown 格式消息到企業微信群。

        Parameters
        ----------
        content : 消息正文（支持企業微信 Markdown 語法）

        Returns
        -------
        bool — True 表示發送成功
        """
        payload = {
            "msgtype": "text",
            "text": {"content": content},
        }
        try:
            resp = requests.post(
                self.webhook_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=10,
            )
            result = resp.json()
            if result.get("errcode") == 0:
                logger.debug("企業微信通知發送成功")
                return True
            else:
                logger.warning(f"企業微信返回錯誤：{result}")
                return False
        except Exception as exc:
            logger.error(f"企業微信通知失敗：{exc}")
            return False

    def send_markdown(self, content: str) -> bool:
        """發送 Markdown 格式消息（支持標題、加粗、顏色等）。"""
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
        try:
            resp = requests.post(
                self.webhook_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                timeout=10,
            )
            result = resp.json()
            return result.get("errcode") == 0
        except Exception as exc:
            logger.error(f"企業微信 Markdown 通知失敗：{exc}")
            return False


# ---------------------------------------------------------------------------
# 郵件通知（SMTP）
# ---------------------------------------------------------------------------

class EmailNotifier:
    """
    通過 SMTP 發送 HTML / 純文本郵件。

    支持 Gmail、QQ 郵箱、企業郵箱等任何標準 SMTP 服務。

    Gmail 配置示例：
        smtp_host: smtp.gmail.com
        smtp_port: 587
        use_tls: true
        username: you@gmail.com
        password: your_app_password   # Gmail 需使用「應用專用密碼」
        from_addr: you@gmail.com
        to_addrs:
          - recipient@example.com
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send(self, subject: str, body: str) -> bool:
        """
        發送純文本郵件。

        Returns
        -------
        bool — True 表示發送成功
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        # 純文本正文
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # HTML 版本（純文本轉 HTML，保留換行）
        html_body = "<pre style='font-family:monospace;'>" + body + "</pre>"
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            if self.use_tls:
                # STARTTLS 模式（端口 587）
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
                server.ehlo()
                server.starttls()
                server.ehlo()
            else:
                # SSL 直連模式（端口 465，QQ 郵箱推薦）
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15)
                server.ehlo()

            server.login(self.username, self.password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            server.quit()
            logger.debug(f"郵件發送成功：{subject}")
            return True
        except smtplib.SMTPAuthenticationError as exc:
            logger.error(f"郵件認證失敗（請確認 QQ 郵箱已開啟 SMTP 並使用授權碼而非登錄密碼）：{exc}")
            return False
        except smtplib.SMTPConnectError as exc:
            logger.error(f"SMTP 連接失敗（請檢查 smtp_host / smtp_port 配置）：{exc}")
            return False
        except Exception as exc:
            logger.error(f"郵件發送失敗：{exc}")
            return False


# ---------------------------------------------------------------------------
# 統一通知入口
# ---------------------------------------------------------------------------

class Notifier:
    """
    統一通知入口，同時管理郵件和企業微信兩個渠道。

    配置示例（config.yaml notify 節）：
    ```yaml
    notify:
      enabled: true
      channels:
        - email
        - wecom

      email:
        smtp_host: smtp.gmail.com
        smtp_port: 587
        use_tls: true
        username: you@gmail.com
        password: ""            # 建議通過環境變量 EMAIL_PASSWORD 傳入
        from_addr: you@gmail.com
        to_addrs:
          - recipient@example.com

      wecom:
        webhook_url: ""         # 企業微信群機器人 Webhook URL
    ```
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled: bool = config.get("enabled", True)
        self.channels: List[str] = config.get("channels", [])

        self._email: Optional[EmailNotifier] = None
        self._wecom: Optional[WeComNotifier] = None

        if "email" in self.channels:
            email_cfg = config.get("email", {})
            if email_cfg.get("smtp_host"):
                import os
                password = (
                    email_cfg.get("password")
                    or os.environ.get("EMAIL_PASSWORD", "")
                )
                self._email = EmailNotifier(
                    smtp_host=email_cfg["smtp_host"],
                    smtp_port=email_cfg.get("smtp_port", 587),
                    username=email_cfg.get("username", ""),
                    password=password,
                    from_addr=email_cfg.get("from_addr", ""),
                    to_addrs=email_cfg.get("to_addrs", []),
                    use_tls=email_cfg.get("use_tls", True),
                )
            else:
                logger.warning("郵件通知已啟用，但 smtp_host 未配置")

        if "wecom" in self.channels:
            wecom_cfg = config.get("wecom", {})
            import os
            webhook = (
                wecom_cfg.get("webhook_url")
                or os.environ.get("WECOM_WEBHOOK_URL", "")
            )
            if webhook:
                self._wecom = WeComNotifier(webhook_url=webhook)
            else:
                logger.warning("企業微信通知已啟用，但 webhook_url 未配置")

    # ------------------------------------------------------------------
    # 業務通知方法
    # ------------------------------------------------------------------

    def send_daily_signals(
        self,
        strategy: str,
        buy: List[str],
        sell: List[str],
        hold: List[str],
        date: Optional[str] = None,
    ) -> None:
        """發送每日信號播報。"""
        if not self.enabled:
            return
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        subject, body = MessageBuilder.daily_signals(strategy, date_str, buy, sell, hold)
        self._dispatch(subject, body)

    def send_stop_loss_alert(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        loss_pct: float,
        threshold_pct: float = 0.08,
    ) -> None:
        """發送止損警報——高優先級，兩個渠道都推送。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.stop_loss_alert(
            symbol, entry_price, current_price, loss_pct, threshold_pct
        )
        logger.warning(f"止損警報：{symbol} 虧損 {loss_pct*100:.2f}%，正在推送通知")
        self._dispatch(subject, body)

    def send_circuit_breaker_alert(
        self,
        drawdown_pct: float,
        halt_threshold_pct: float,
        equity: float,
    ) -> None:
        """發送熔斷警報。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.circuit_breaker_alert(
            drawdown_pct, halt_threshold_pct, equity
        )
        logger.error(f"熔斷警報：組合回撤 {drawdown_pct*100:.2f}%，推送通知")
        self._dispatch(subject, body)

    def send_buy_signal_alert(
        self,
        symbol: str,
        price: float,
        signal_type: str,
        short_detail: Optional[Dict[str, Any]] = None,
        long_detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """發送買入信號通知（短線 / 長線 / 兩者均觸發）。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.buy_signal_alert(
            symbol=symbol,
            price=price,
            signal_type=signal_type,
            short_detail=short_detail or {},
            long_detail=long_detail or {},
        )
        logger.info(f"買入信號：{symbol} ({signal_type})，推送通知")
        self._dispatch(subject, body)

    def send_buy_signal_digest(
        self,
        scan_count: int,
        rows: List[Dict[str, Any]],
        detail_limit: int = 8,
    ) -> None:
        """全池掃描後多只股票合併為一封通知。"""
        if not self.enabled or not rows:
            return
        subject, body = MessageBuilder.buy_signal_digest(
            scan_count=scan_count,
            rows=rows,
            detail_limit=detail_limit,
        )
        logger.info(f"買入信號匯總：掃描 {scan_count} 只，觸發 {len(rows)} 只，推送一封匯總郵件")
        self._dispatch(subject, body)

    def send_portfolio_report(self, summary: Dict[str, Any]) -> None:
        """發送每日收盤持倉報告。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.portfolio_report(summary)
        self._dispatch(subject, body)

    def send_raw(self, subject: str, body: str) -> None:
        """發送自定義消息（用於調試或自定義場景）。"""
        if not self.enabled:
            return
        self._dispatch(subject, body)

    def send_weekly_report(self, summary: Dict[str, Any]) -> None:
        """發送週度持倉報告（每週五收盤後）。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.weekly_report(summary)
        logger.info(f"週報推送：{summary.get('period', '本週')}")
        self._dispatch(subject, body)

    def send_risk_dashboard(self, dashboard: Dict[str, Any]) -> None:
        """發送風控儀表板摘要（隨每日報告一起發送）。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.risk_dashboard_alert(dashboard)
        self._dispatch(subject, body)

    def send_black_swan_alert(
        self,
        return_pct: float,
        reason: str,
        action: str = "緊急縮倉",
        equity: float = 0.0,
    ) -> None:
        """發送黑天鵝緊急警報——最高優先級。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.black_swan_alert(return_pct, reason, action, equity)
        logger.error(f"黑天鵝警報推送：{reason[:80]}")
        self._dispatch(subject, body)

    def send_trade_executed(
        self,
        orders: List[Dict[str, Any]],
        regime: str = "",
    ) -> None:
        """信號觸發後通知實際下單明細。"""
        if not self.enabled or not orders:
            return
        subject, body = MessageBuilder.trade_executed(orders, regime)
        logger.info(f"下單通知：{len(orders)} 筆訂單，market_regime={regime}")
        self._dispatch(subject, body)

    def send_daily_review(self, review: Dict[str, Any]) -> None:
        """发送收盘后自动复盘日报（结论先行）。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.daily_review(review)
        self._dispatch(subject, body)

    def send_weekly_strategy_upgrade(self, payload: Dict[str, Any]) -> None:
        """发送周度策略升级总结（结论 + 建议）。"""
        if not self.enabled:
            return
        subject, body = MessageBuilder.weekly_strategy_upgrade(payload)
        self._dispatch(subject, body)

    # ------------------------------------------------------------------
    # 內部分發
    # ------------------------------------------------------------------

    def _dispatch(self, subject: str, body: str) -> None:
        """向所有已配置渠道分發消息。"""
        success_count = 0

        if self._email:
            if self._email.send(subject, body):
                success_count += 1

        if self._wecom:
            # 企業微信消息前綴添加主題行，方便快速識別
            wecom_body = f"**{subject}**\n\n{body}"
            if self._wecom.send(wecom_body):
                success_count += 1

        if success_count == 0 and (self._email or self._wecom):
            logger.error(f"所有通知渠道均發送失敗：{subject}")
        else:
            logger.info(f"通知已發送至 {success_count} 個渠道：{subject}")
