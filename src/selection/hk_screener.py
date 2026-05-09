"""
港股（HK market）前置過濾器

在進入策略信號計算之前，過濾掉不適合短線交易的標的：
1. 停牌（最新價為 0 或 NaN）
2. 股價 < 1 HKD（仙股 / 細價股）
3. 流通市值超出 [50 億, 2000 億] HKD
   ↳ 流通市值由快照數據推算：成交量 ÷ (換手率/100) × 最新價
4. 換手率超出 [0.5%, 8%]
5. 負資產標的（市淨率 < 0，即 P/B < 0 → 股東權益為負）

Usage
-----
snapshot = ak.stock_hk_spot_em()   # or stock_hk_main_board_spot_em()
screener = HKScreener(config.get("screener", {}).get("hk", {}))
passing  = screener.apply_hk(prices, snapshot)

akshare stock_hk_spot_em() 欄位對照
────────────────────────────────────
代码       股票代碼（5 位純數字）
名称       股票名稱
最新价     最新成交價（HKD）
涨跌幅     當日漲跌幅（%）
成交量     當日成交股數
成交额     當日成交金額（HKD）
换手率     換手率（%）
市净率     市淨率 P/B（負值 = 負資產）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from .screener import Screener


class HKScreener(Screener):
    """
    港股專用前置過濾器，繼承通用 Screener 並疊加 HK 特有規則。

    Parameters
    ----------
    config : dict — 可覆蓋以下默認值：
        price_min           float  股價下限（HKD）         default 1.0
        float_cap_min       float  流通市值下限（HKD）     default 50e8  (50 億)
        float_cap_max       float  流通市值上限（HKD）     default 2000e8 (2000 億)
        turnover_min        float  換手率下限（%）         default 0.5
        turnover_max        float  換手率上限（%）         default 8.0
        exclude_negative_pb bool   排除負資產（P/B < 0）   default True
    """

    _HK_DEFAULTS: Dict[str, Any] = {
        "price_min": 1.0,
        "float_cap_min": 50e8,
        "float_cap_max": 2000e8,
        "turnover_min": 0.5,
        "turnover_max": 8.0,
        "exclude_negative_pb": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**self._HK_DEFAULTS, **(config or {})}
        super().__init__(
            config={
                "min_price": merged["price_min"],
                "min_avg_dollar_volume": 0,
                "min_listing_days": 60,
                "exclude_sectors": [],
            }
        )
        self.hk_config = merged

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def apply_hk(
        self,
        prices: pd.DataFrame,
        snapshot: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        執行港股前置過濾。

        Parameters
        ----------
        prices   : MultiIndex (symbol, field) × date — 日線歷史數據
        snapshot : akshare stock_hk_spot_em() 返回的 DataFrame
                   需包含欄位：代码、最新价、换手率、成交量、市净率
                   為 None 時跳過快照依賴的過濾

        Returns
        -------
        list of symbol strings that pass all filters
        """
        from ..data.preprocessor import Preprocessor

        close = Preprocessor.get_close(prices)
        all_syms: List[str] = close.columns.tolist()
        logger.info(f"HKScreener: 起始股票池 {len(all_syms)} 只")

        snap_pass = all_syms
        if snapshot is not None and not snapshot.empty:
            snap_pass = self._filter_by_snapshot(all_syms, snapshot)
        else:
            logger.warning("HKScreener: 未提供快照，跳過基本面快照過濾")

        logger.info(f"HKScreener: 最終通過 {len(snap_pass)} 只")
        return snap_pass

    # ------------------------------------------------------------------
    # 私有過濾方法
    # ------------------------------------------------------------------

    def _filter_by_snapshot(
        self,
        symbols: List[str],
        snapshot: pd.DataFrame,
    ) -> List[str]:
        """
        依賴 akshare 快照，過濾：
        - 停牌（最新價 = 0 / NaN）
        - 股價 < price_min（仙股）
        - 流通市值超出範圍（由成交量 + 換手率 + 最新價推算）
        - 換手率超出範圍
        - 負資產（市淨率 < 0）
        """
        cfg = self.hk_config

        code_col = self._detect_col(snapshot, ["代码", "代碼", "symbol"])
        price_col = self._detect_col(snapshot, ["最新价", "最新價", "price"])
        vol_col = self._detect_col(snapshot, ["成交量", "volume"])
        turnover_col = self._detect_col(snapshot, ["换手率", "換手率", "turnover_rate"])
        pb_col = self._detect_col(snapshot, ["市净率", "市淨率", "pb_ratio"])

        if code_col is None:
            logger.warning("HKScreener: 快照缺少代码列，跳過快照過濾")
            return symbols

        snap = snapshot.copy()
        # HK 代碼通常是 5 位純數字，統一填充前導 0
        snap[code_col] = snap[code_col].astype(str).str.zfill(5)
        snap = snap.set_index(code_col)

        result = []
        halted = price_low = cap_excluded = turnover_excluded = neg_asset = 0

        for sym in symbols:
            # 代碼標準化：去掉後綴（如 .HK），取純數字部分並填充 5 位
            code = sym.split(".")[0].lstrip("0") if "." in sym else sym.lstrip("0")
            code = code.zfill(5)

            if code not in snap.index:
                result.append(sym)  # 快照無此股 → 保留（避免遺漏）
                continue

            row = snap.loc[code]

            # ── 1. 停牌（最新價為空或 0）──────────────────────────────
            price = pd.to_numeric(
                row.get(price_col) if price_col else None, errors="coerce"
            )
            if pd.isna(price) or price <= 0:
                halted += 1
                continue

            # ── 2. 仙股（股價 < price_min）────────────────────────────
            if price < cfg["price_min"]:
                price_low += 1
                continue

            # ── 3. 負資產（市淨率 < 0）────────────────────────────────
            if cfg["exclude_negative_pb"] and pb_col is not None:
                pb = pd.to_numeric(row.get(pb_col), errors="coerce")
                if pd.notna(pb) and pb < 0:
                    neg_asset += 1
                    continue

            # ── 4. 換手率篩選 ──────────────────────────────────────────
            turnover = None
            if turnover_col is not None:
                turnover = pd.to_numeric(row.get(turnover_col), errors="coerce")
                if pd.notna(turnover):
                    if turnover < cfg["turnover_min"] or turnover > cfg["turnover_max"]:
                        turnover_excluded += 1
                        continue

            # ── 5. 流通市值篩選（由成交量 + 換手率 + 最新價推算）─────
            if vol_col is not None and turnover is not None and pd.notna(turnover) and turnover > 0:
                volume = pd.to_numeric(row.get(vol_col), errors="coerce")
                if pd.notna(volume) and volume > 0:
                    # 流通股數 = 成交量 / (換手率 / 100)
                    # 流通市值 = 流通股數 × 最新價
                    float_shares = volume / (turnover / 100.0)
                    float_cap = float_shares * price
                    if float_cap < cfg["float_cap_min"] or float_cap > cfg["float_cap_max"]:
                        cap_excluded += 1
                        continue

            result.append(sym)

        logger.debug(
            f"HK快照過濾: 停牌 {halted}, 仙股 {price_low}, 負資產 {neg_asset}, "
            f"換手率排除 {turnover_excluded}, 市值排除 {cap_excluded}, "
            f"通過 {len(result)}"
        )
        return result

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def summary(self) -> Dict[str, Any]:
        base = super().summary()
        base.update({"hk_config": self.hk_config})
        return base
