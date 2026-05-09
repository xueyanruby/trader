"""
Futu 模擬盤交易 Broker（TrdEnv.SIMULATE）

通過 Futu OpenD 的 OpenTradeContext 接口，在官方模擬環境中下單、查倉、查資金。
所有調用均使用 TrdEnv.SIMULATE，確保不影響真實賬戶。

前置條件
--------
1. 本地已啟動 FutuOpenD（默認地址 127.0.0.1:11111）
2. pip install futu-api>=10.4

市場支持
--------
- "hk"    : 港股（HK.*）—— 限價單（OrderType.NORMAL），以快照價 +/- 0.3% 滑點模擬市價
- "us"    : 美股（US.*）—— 市價單（OrderType.MARKET）
- "multi" : 按代碼自動識別 HK / US / CN

符號格式（內部 → Futu OpenD）
------------------------------
  "00700"      → "HK.00700"
  "AAPL"       → "US.AAPL"
  "600519"     → "SH.600519"（6 開頭）
  "000001"     → "SZ.000001"（非 6 開頭）
  "HK.00700"   → 直接透傳

注意：Futu 模擬盤手數
- 港股最小手數通常為 100 或 500，本實現默認 100 股取整
- 如需精確手數，可通過 Futu get_stock_basicinfo 查詢

使用示例
--------
broker = FutuTradeBroker(host="127.0.0.1", port=11111, market="hk")
broker.update_prices({"00700": 380.0})
order = broker.place_order("00700", qty=100, side="buy")
summary = broker.portfolio_summary()
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from .base_broker import BaseBroker, Order, Position

# ---------------------------------------------------------------------------
# 懶加載 futu-api（允許在未安裝時仍可 import 本模塊）
# ---------------------------------------------------------------------------
try:
    from futu import (  # type: ignore
        KLType,
        OpenSecTradeContext,
        OrderStatus,
        OrderType,
        RET_OK,
        TrdEnv,
        TrdMarket,
        TrdSide,
    )
    _FUTU_AVAILABLE = True
except ImportError:
    _FUTU_AVAILABLE = False
    OpenSecTradeContext = None  # type: ignore
    TrdEnv = None           # type: ignore
    TrdMarket = None        # type: ignore
    TrdSide = None          # type: ignore
    OrderType = None        # type: ignore
    OrderStatus = None      # type: ignore
    RET_OK = 0              # type: ignore
    logger.warning(
        "futu-api 未安裝或版本不兼容，FutuTradeBroker 不可用。\n"
        "請執行：pip install futu-api\n"
        "（若已安裝仍報錯，請確認 futu-api 版本是否支持 OpenSecTradeContext）"
    )


# HK 股默認手數（lot size）
_DEFAULT_LOT_SIZE = 100


def _to_futu_code(symbol: str, default_market: str = "US") -> str:
    """
    將內部符號轉換為 Futu OpenD 格式（市場前綴.代碼）。

    支持的輸入：
      "00700"       → "HK.00700"
      "00700.HK"    → "HK.00700"
      "AAPL"        → "US.AAPL"
      "600519"      → "SH.600519"
      "000001"      → "SZ.000001"
      "HK.00700"    → "HK.00700"（直接透傳）
    """
    s = symbol.strip()
    upper = s.upper()

    # 已是 Futu 格式
    for prefix in ("HK.", "SH.", "SZ.", "US."):
        if upper.startswith(prefix):
            return f"{prefix[:-1]}.{upper[len(prefix):]}"

    # yfinance 格式：xxxx.HK
    if upper.endswith(".HK"):
        code = upper[:-3].zfill(5)
        return f"HK.{code}"

    # 純數字：5位 → HK，6位 → CN
    if s.isdigit():
        if len(s) == 5:
            return f"HK.{s.zfill(5)}"
        if len(s) == 6:
            market = "SH" if s.startswith("6") else "SZ"
            return f"{market}.{s}"

    # 字母 → US（默認）或 default_market
    return f"{default_market}.{upper}"


def _futu_code_to_original(futu_code: str) -> str:
    """Futu 格式 → 內部符號（去掉市場前綴）。"""
    if "." in futu_code:
        parts = futu_code.split(".", 1)
        prefix = parts[0].upper()
        code = parts[1]
        if prefix == "HK":
            return code.lstrip("0").zfill(5) if code.isdigit() else code
        return code
    return futu_code


def _round_to_lot(qty: float, lot_size: int = _DEFAULT_LOT_SIZE) -> int:
    """向下取整到最近的手數倍數，最少 1 手。"""
    lots = int(qty) // lot_size
    return max(lots, 1) * lot_size


class FutuTradeBroker(BaseBroker):
    """
    通過 Futu OpenD 的 TrdEnv.SIMULATE 進行模擬盤交易。

    Parameters
    ----------
    host      : OpenD 主機（默認 127.0.0.1）
    port      : OpenD 端口（默認 11111）
    market    : 主要市場 "hk" | "us" | "multi"，影響下單時的 TrdMarket 選擇
    acc_index : 模擬賬戶索引（默認 0）
    lot_size  : HK 股手數（默認 100）
    slippage_bps : HK 限價單滑點（基點，默認 30 = 0.3%）
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        market: str = "hk",
        acc_index: int = 0,
        lot_size: int = _DEFAULT_LOT_SIZE,
        slippage_bps: float = 30.0,
    ) -> None:
        if not _FUTU_AVAILABLE:
            raise ImportError(
                "futu-api 未安裝或版本不兼容，無法使用 FutuTradeBroker。\n"
                "請執行：pip install futu-api\n"
                "（若已安裝仍報錯，請確認 futu-api 版本是否支持 OpenSecTradeContext）"
            )

        self._host = host
        self._port = port
        self._market_cfg = market.lower()
        self._acc_index = acc_index
        self._lot_size = lot_size
        self._slippage = slippage_bps / 10_000

        # TrdMarket 映射
        self._trd_market = {
            "hk": TrdMarket.HK,
            "us": TrdMarket.US,
            # futu-api 新版本使用 CN（舊版本可能叫 MAINLAND）
            "cn": getattr(TrdMarket, "CN", TrdMarket.US),
        }.get(market.lower(), TrdMarket.HK)

        # 價格緩存（由 update_prices() 注入，供 get_latest_price 和下單使用）
        self._price_cache: Dict[str, float] = {}

        # 本地訂單快取（用於 weekly_fill_log 等報告）
        self._fill_log: List[Dict[str, Any]] = []

        # 建立交易上下文
        self._trd_ctx = OpenSecTradeContext(host=host, port=port)

        logger.info(
            f"FutuTradeBroker 初始化：host={host}:{port}  "
            f"market={market}  acc_index={acc_index}  "
            f"env=SIMULATE  lot_size={lot_size}"
        )

    # ------------------------------------------------------------------
    # 生命週期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """關閉與 OpenD 的交易連接。"""
        try:
            if self._trd_ctx is not None:
                self._trd_ctx.close()
                self._trd_ctx = None  # type: ignore[assignment]
                logger.info("FutuTradeBroker: 交易連接已關閉")
        except Exception as exc:
            logger.warning(f"FutuTradeBroker.close() 異常：{exc}")

    # ------------------------------------------------------------------
    # 賬戶資金
    # ------------------------------------------------------------------

    def get_cash(self) -> float:
        """返回模擬賬戶可用現金。"""
        try:
            ret, data = self._trd_ctx.accinfo_query(
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
            )
            if ret != RET_OK:
                logger.warning(f"get_cash 失敗：{data}")
                return 0.0
            # 列名：cash / available_funds / power（視賬戶類型）
            for col in ("cash", "available_funds", "power", "cash_balance"):
                if col in data.columns:
                    return float(data[col].iloc[0])
            # 後備：取第一列數值
            return float(data.iloc[0, 0])
        except Exception as exc:
            logger.error(f"get_cash 異常：{exc}")
            return 0.0

    def get_equity(self) -> float:
        """返回模擬賬戶總資產淨值（現金 + 持倉市值）。"""
        try:
            ret, data = self._trd_ctx.accinfo_query(
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
            )
            if ret != RET_OK:
                logger.warning(f"get_equity 失敗：{data}")
                return 0.0
            for col in ("total_assets", "net_asset_val", "market_val", "total_asset"):
                if col in data.columns:
                    return float(data[col].iloc[0])
            # 後備：返回 cash
            return self.get_cash()
        except Exception as exc:
            logger.error(f"get_equity 異常：{exc}")
            return 0.0

    # ------------------------------------------------------------------
    # 持倉查詢
    # ------------------------------------------------------------------

    def get_positions(self) -> Dict[str, Position]:
        """返回模擬賬戶所有持倉 {内部符号: Position}。"""
        try:
            ret, data = self._trd_ctx.position_list_query(
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
            )
            if ret != RET_OK:
                logger.warning(f"get_positions 失敗：{data}")
                return {}
            if data is None or data.empty:
                return {}

            positions: Dict[str, Position] = {}
            for _, row in data.iterrows():
                try:
                    futu_code = str(row.get("code", ""))
                    original = _futu_code_to_original(futu_code)
                    qty = float(row.get("qty", 0))
                    avg_cost = float(row.get("cost_price", row.get("avg_cost", 0)))
                    current_price = self._price_cache.get(original, float(row.get("market_price", avg_cost)))
                    if qty > 0:
                        positions[original] = Position(
                            symbol=original,
                            qty=qty,
                            avg_cost=avg_cost,
                            current_price=current_price,
                        )
                except Exception as exc:
                    logger.debug(f"解析持倉行失敗：{exc}")
            return positions
        except Exception as exc:
            logger.error(f"get_positions 異常：{exc}")
            return {}

    # ------------------------------------------------------------------
    # 下單 / 撤單
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """
        在 Futu 模擬盤下單。

        HK 股：使用限價單（NORMAL），以快照價 +/- slippage 作為限價
        US 股：使用市價單（MARKET），可跳過 limit_price
        """
        futu_code = _to_futu_code(symbol, default_market="US")
        market_prefix = futu_code.split(".")[0].upper()

        trd_side = TrdSide.BUY if side.lower() == "buy" else TrdSide.SELL

        # 手數取整（HK）
        is_hk = market_prefix == "HK"
        if is_hk:
            qty = _round_to_lot(qty, self._lot_size)

        # 確定下單價格
        snapshot_price = self._price_cache.get(symbol, 0.0)
        if limit_price is not None:
            price = float(limit_price)
        elif snapshot_price > 0:
            if is_hk:
                # HK 用快照價 + 滑點模擬市價
                slip = self._slippage if side.lower() == "buy" else -self._slippage
                price = round(snapshot_price * (1 + slip), 3)
            else:
                # US 用 0（市價單不需要 price）
                price = 0.0
        else:
            logger.warning(f"FutuTradeBroker: {symbol} 無價格緩存，訂單可能被拒絕")
            price = 0.0

        # Futu 訂單類型
        if is_hk or order_type == "limit":
            futu_order_type = OrderType.NORMAL  # HK 限價
        else:
            futu_order_type = OrderType.MARKET  # US 市價

        order_id = str(uuid.uuid4())[:8]  # 臨時本地 ID，submit 後替換

        try:
            ret, data = self._trd_ctx.place_order(
                price=price,
                qty=qty,
                code=futu_code,
                trd_side=trd_side,
                order_type=futu_order_type,
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
                remark=f"auto:{order_id}",
            )

            if ret != RET_OK:
                logger.error(f"place_order 失敗 [{symbol}]：{data}")
                order = Order(
                    symbol=symbol, qty=qty, side=side.lower(),
                    order_type=order_type, limit_price=price or None,
                    order_id=order_id, status="rejected",
                    submitted_at=datetime.utcnow(),
                )
                return order

            # 從返回數據取 Futu 訂單號
            futu_order_id = str(data["order_id"].iloc[0]) if "order_id" in data.columns else order_id
            logger.info(
                f"[SIMULATE] {side.upper():4s} {qty} {symbol} "
                f"@ {'mkt' if price == 0 else f'{price:.3f}'}  "
                f"futu_id={futu_order_id}"
            )

            order = Order(
                symbol=symbol,
                qty=qty,
                side=side.lower(),
                order_type=order_type,
                limit_price=price if price > 0 else None,
                order_id=futu_order_id,
                status="pending",
                submitted_at=datetime.utcnow(),
            )

            # 記錄到本地成交日誌（等後續查詢狀態後更新）
            self._fill_log.append({
                "order_id": futu_order_id,
                "symbol": symbol,
                "futu_code": futu_code,
                "side": side.lower(),
                "qty": qty,
                "price": price,
                "status": "pending",
                "submitted_at": order.submitted_at,
            })

            return order

        except Exception as exc:
            logger.error(f"place_order 異常 [{symbol}]：{exc}")
            return Order(
                symbol=symbol, qty=qty, side=side.lower(),
                order_type=order_type, order_id=order_id,
                status="rejected", submitted_at=datetime.utcnow(),
            )

    def cancel_order(self, order_id: str) -> bool:
        """撤銷模擬盤訂單。"""
        try:
            ret, data = self._trd_ctx.cancel_order(
                orderid=order_id,
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
            )
            if ret == RET_OK:
                logger.info(f"[SIMULATE] 撤單成功：order_id={order_id}")
                return True
            logger.warning(f"cancel_order 失敗：{data}")
            return False
        except Exception as exc:
            logger.error(f"cancel_order 異常：{exc}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """查詢單個訂單狀態。"""
        orders = self._query_order_list(order_id=order_id)
        return orders[0] if orders else None

    def get_open_orders(self) -> List[Order]:
        """返回所有未成交掛單。"""
        return self._query_order_list(status_filter=[
            OrderStatus.SUBMITTING,
            OrderStatus.SUBMITTED,
            OrderStatus.FILLED_PART,
        ])

    # ------------------------------------------------------------------
    # 價格緩存（供止損監控和下單使用）
    # ------------------------------------------------------------------

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        注入最新市場價格（與 PaperBroker 同接口）。
        由 scheduler._run_price_monitor 在每輪快照後調用。
        """
        self._price_cache.update(prices)

    def get_latest_price(self, symbol: str) -> float:
        """從本地價格緩存取最新價。"""
        return self._price_cache.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # 報告
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> Dict[str, Any]:
        """
        返回與 PaperBroker.portfolio_summary() 同 schema 的快照。
        供 notifier.send_portfolio_report() 使用。
        """
        try:
            cash = self.get_cash()
            equity = self.get_equity()
            positions = self.get_positions()

            pos_data: Dict[str, Any] = {}
            for sym, pos in positions.items():
                current_price = self._price_cache.get(sym, pos.current_price)
                pos.current_price = current_price
                pos_data[sym] = {
                    "qty": pos.qty,
                    "avg_cost": pos.avg_cost,
                    "current_price": current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": f"{pos.unrealized_pnl_pct * 100:.2f}%",
                }

            return {
                "cash": cash,
                "equity": equity,
                "num_positions": len(positions),
                "positions": pos_data,
                "broker": "futu_simulate",
            }
        except Exception as exc:
            logger.error(f"portfolio_summary 異常：{exc}")
            return {"cash": 0.0, "equity": 0.0, "num_positions": 0, "positions": {}}

    def fill_log(self) -> List[Dict[str, Any]]:
        """返回本地訂單快取（含提交狀態）。"""
        return list(self._fill_log)

    def weekly_fill_log(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        查詢一週的成交記錄，計算週 P&L。

        Parameters
        ----------
        start_date : 週一（默認：本週一）
        end_date   : 週五（默認：今天）

        Returns
        -------
        {
          "period"       : "2025-05-05 ~ 2025-05-09",
          "fills"        : [ {order_id, symbol, side, qty, price, ...} ],
          "total_trades" : int,
          "start_equity" : float,   # 近似，來自賬戶 (暫時用 equity)
          "end_equity"   : float,
          "return_pct"   : float,
          "top_buys"     : [symbol, ...],
          "top_sells"    : [symbol, ...],
        }
        """
        today = date.today()
        if end_date is None:
            end_date = today
        if start_date is None:
            days_since_monday = today.weekday()
            start_date = today - timedelta(days=days_since_monday)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        fills: List[Dict[str, Any]] = []
        try:
            ret, data = self._trd_ctx.history_order_list_query(
                status_filter_list=[OrderStatus.FILLED_ALL, OrderStatus.FILLED_PART],
                start=start_str,
                end=end_str,
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
            )
            if ret == RET_OK and data is not None and not data.empty:
                for _, row in data.iterrows():
                    try:
                        futu_code = str(row.get("code", ""))
                        original = _futu_code_to_original(futu_code)
                        fills.append({
                            "order_id": str(row.get("order_id", "")),
                            "symbol": original,
                            "side": str(row.get("trd_side", "")).lower(),
                            "qty": float(row.get("dealt_qty", row.get("qty", 0))),
                            "price": float(row.get("dealt_avg_price", row.get("price", 0))),
                            "create_time": str(row.get("create_time", "")),
                        })
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(f"weekly_fill_log 查詢 Futu 失敗，回退本地緩存：{exc}")
            # 回退：過濾本地 fill_log
            for entry in self._fill_log:
                sub_at = entry.get("submitted_at")
                if sub_at:
                    sub_date = sub_at.date() if hasattr(sub_at, "date") else sub_at
                    if start_date <= sub_date <= end_date:
                        fills.append(entry)

        equity = self.get_equity()
        top_buys = list({f["symbol"] for f in fills if f.get("side") == "buy"})[:5]
        top_sells = list({f["symbol"] for f in fills if f.get("side") == "sell"})[:5]

        return {
            "period": f"{start_str} ~ {end_str}",
            "fills": fills,
            "total_trades": len(fills),
            "end_equity": equity,
            "start_equity": equity,   # 精確值需外部傳入
            "return_pct": 0.0,        # 精確值需外部傳入
            "top_buys": top_buys,
            "top_sells": top_sells,
        }

    # ------------------------------------------------------------------
    # 私有輔助
    # ------------------------------------------------------------------

    def _query_order_list(
        self,
        order_id: str = "",
        status_filter: Optional[List] = None,
    ) -> List[Order]:
        """通用訂單查詢，返回 Order 列表。"""
        try:
            sf = status_filter or []
            ret, data = self._trd_ctx.order_list_query(
                order_id=order_id,
                status_filter_list=sf,
                trd_env=TrdEnv.SIMULATE,
                acc_index=self._acc_index,
            )
            if ret != RET_OK or data is None or data.empty:
                return []

            orders: List[Order] = []
            for _, row in data.iterrows():
                try:
                    futu_code = str(row.get("code", ""))
                    original = _futu_code_to_original(futu_code)
                    status_raw = str(row.get("order_status", "")).lower()
                    status = (
                        "filled" if "filled_all" in status_raw else
                        "pending" if any(s in status_raw for s in ("submit", "wait")) else
                        "cancelled" if "cancel" in status_raw else
                        status_raw
                    )
                    side_raw = str(row.get("trd_side", "")).lower()
                    side = "buy" if "buy" in side_raw else "sell"
                    orders.append(Order(
                        symbol=original,
                        qty=float(row.get("qty", 0)),
                        side=side,
                        order_type="limit",
                        limit_price=float(row.get("price", 0)) or None,
                        order_id=str(row.get("order_id", "")),
                        status=status,
                        filled_qty=float(row.get("dealt_qty", 0)),
                        filled_avg_price=float(row.get("dealt_avg_price", 0)) or None,
                        submitted_at=None,
                    ))
                except Exception as exc:
                    logger.debug(f"解析訂單行失敗：{exc}")
            return orders
        except Exception as exc:
            logger.error(f"_query_order_list 異常：{exc}")
            return []
