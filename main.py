"""
main.py — CLI entry point for the auto trader system.

Commands
--------
backtest   Run a historical backtest for the configured strategy.
paper      Run paper trading using live market data and the PaperBroker.
list       List all registered strategies.
info       Show current configuration.

Usage examples
--------------
# Run a backtest with default config
python main.py backtest

# Run a backtest with a specific strategy
python main.py backtest --strategy momentum_12_1

# Use a custom config file and date range
python main.py backtest --config config/my_config.yaml --start 2020-01-01 --end 2023-12-31

# Run paper trading (uses today's data)
python main.py paper --strategy multi_factor

# Show all registered strategies
python main.py list

# Print effective configuration
python main.py info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

# 自動加載項目根目錄的 .env 文件（若存在），使 EMAIL_PASSWORD 等憑據無需每次手動 export
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is importable regardless of working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation="10 MB", retention="30 days")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {path}. Using defaults.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Command: backtest
# ---------------------------------------------------------------------------

def cmd_backtest(args: argparse.Namespace, config: dict) -> None:
    """Run a historical backtest."""
    from src.data.us_fetcher import USFetcher
    from src.data.cn_hk_fetcher import CNHKFetcher
    from src.data.futu_cnhk_fetcher import FutuCNHKFetcher
    from src.data.futu_us_fetcher import FutuUSFetcher
    from src.data.preprocessor import Preprocessor
    from src.selection.screener import Screener
    from src.backtest.engine import BacktestEngine
    from src.strategies import list_strategies

    strategy_name = args.strategy or config.get("strategy", {}).get("active", "momentum_12_1")
    start = args.start or config.get("data", {}).get("start_date", "2018-01-01")
    end = args.end or config.get("data", {}).get("end_date", "2024-12-31")
    market_mode = config.get("market", {}).get("mode", "us")

    logger.info(f"=== BACKTEST: strategy={strategy_name} | market={market_mode} | {start} → {end} ===")

    # --- Fetch data ---
    data_cfg = config.get("data", {})
    provider_cfg = data_cfg.get("provider", {}) or {}
    if market_mode in ("us", "multi"):
        us_provider = provider_cfg.get("us", "yfinance")
        universe_fetcher = USFetcher(
            cache_dir=data_cfg.get("cache_dir", ".cache/data"),
            cache_enabled=data_cfg.get("cache_enabled", True),
        )
        if us_provider == "futu":
            fetcher = FutuUSFetcher(
                cache_dir=data_cfg.get("cache_dir", ".cache/data"),
                cache_enabled=data_cfg.get("cache_enabled", True),
            )
        else:
            fetcher = universe_fetcher
        universe = config.get("market", {}).get("us", {}).get("universe") or []
        if not universe:
            logger.info("No universe configured — fetching S&P 500")
            universe = universe_fetcher.get_universe()

        # Limit universe size for speed during development
        if args.fast and len(universe) > 50:
            logger.info(f"--fast mode: limiting universe to 50 symbols")
            universe = universe[:50]

        prices = fetcher.get_prices(universe, start=start, end=end)
        benchmark_symbol = config.get("market", {}).get("us", {}).get("benchmark", "SPY")
        benchmark_prices = fetcher.get_prices([benchmark_symbol], start=start, end=end)

    elif market_mode == "cn":
        cn_provider = provider_cfg.get("cn", "futu")
        if cn_provider == "akshare":
            fetcher = CNHKFetcher(market="cn", cache_dir=data_cfg.get("cache_dir", ".cache/data"))
        else:
            fetcher = FutuCNHKFetcher(market="cn", cache_dir=data_cfg.get("cache_dir", ".cache/data"))
        universe = config.get("market", {}).get("cn", {}).get("universe") or []
        if not universe:
            universe = fetcher.get_universe()
        if args.fast and len(universe) > 50:
            universe = universe[:50]
        prices = fetcher.get_prices(universe, start=start, end=end)
        benchmark_prices = None

    elif market_mode == "hk":
        hk_provider = provider_cfg.get("hk", "futu")
        if hk_provider == "akshare":
            fetcher = CNHKFetcher(market="hk", cache_dir=data_cfg.get("cache_dir", ".cache/data"))
        else:
            fetcher = FutuCNHKFetcher(market="hk", cache_dir=data_cfg.get("cache_dir", ".cache/data"))
        universe = config.get("market", {}).get("hk", {}).get("universe") or []
        if not universe:
            universe = fetcher.get_universe()
        if args.fast and len(universe) > 50:
            universe = universe[:50]
        prices = fetcher.get_prices(universe, start=start, end=end)
        benchmark_prices = None
    else:
        logger.error(f"Unknown market mode: {market_mode}")
        sys.exit(1)

    if prices.empty:
        logger.error("No price data fetched. Check your data config and internet connection.")
        sys.exit(1)

    # --- Clean data ---
    prices = Preprocessor.clean(prices)

    # --- Screen universe ---
    screener = Screener(config.get("screener", {}))
    screened = screener.apply(prices)

    # --- Run backtest ---
    engine = BacktestEngine(config)
    result = engine.run(
        prices=prices,
        strategy_name=strategy_name,
        benchmark_prices=benchmark_prices,
        screened_symbols=screened,
    )

    # --- Plot ---
    if not args.no_plot:
        result.plot()

    logger.info("Backtest complete.")


# ---------------------------------------------------------------------------
# Command: paper trading
# ---------------------------------------------------------------------------

def cmd_paper(args: argparse.Namespace, config: dict) -> None:
    """Run a paper trading session using the latest available data."""
    import time
    from datetime import date
    from src.data.us_fetcher import USFetcher
    from src.data.futu_us_fetcher import FutuUSFetcher
    from src.data.preprocessor import Preprocessor
    from src.selection.screener import Screener
    from src.strategies import get_strategy
    from src.risk.manager import RiskManager

    strategy_name = args.strategy or config.get("strategy", {}).get("active", "momentum_12_1")
    data_cfg = config.get("data", {})
    risk_cfg = config.get("risk", {})

    broker = _build_broker(config)
    risk_manager = RiskManager(risk_cfg)

    logger.info(f"=== PAPER TRADING: strategy={strategy_name} ===")
    logger.info(f"Starting equity: ${broker.get_equity():,.2f}")

    # Fetch recent history for signal generation
    today = str(date.today())
    start = data_cfg.get("start_date", "2020-01-01")

    universe_fetcher = USFetcher(
        cache_dir=data_cfg.get("cache_dir", ".cache/data"),
        cache_enabled=data_cfg.get("cache_enabled", True),
    )
    us_provider = (data_cfg.get("provider", {}) or {}).get("us", "yfinance")
    if us_provider == "futu":
        fetcher = FutuUSFetcher(
            cache_dir=data_cfg.get("cache_dir", ".cache/data"),
            cache_enabled=data_cfg.get("cache_enabled", True),
        )
    else:
        fetcher = universe_fetcher

    universe = universe_fetcher.get_universe()
    if args.fast:
        universe = universe[:50]

    prices = fetcher.get_prices(universe, start=start, end=today)
    prices = Preprocessor.clean(prices)

    screener = Screener(config.get("screener", {}))
    screened = screener.apply(prices)

    # Generate signals
    strategy_cls = get_strategy(strategy_name)
    strategy_cfg = config.get("strategy", {}).get(strategy_name, {})
    strategy = strategy_cls(config=strategy_cfg)
    signals = strategy.generate_signals(prices[screened] if screened else prices)

    if signals.empty:
        logger.warning("No signals generated. Nothing to trade.")
        return

    # Get latest prices for sizing
    close = Preprocessor.get_close(prices)
    latest_prices = close.iloc[-1].to_dict()
    broker.update_prices(latest_prices)

    # Size positions
    target_weights = risk_manager.size_positions(
        signals=signals,
        prices=close.iloc[-1],
    )

    # Execute rebalance
    current_prices = {sym: latest_prices.get(sym, 0.0) for sym in target_weights}
    orders = broker.rebalance(target_weights, current_prices)

    logger.info(f"Placed {len(orders)} orders")
    for order in orders:
        logger.info(
            f"  {order.side.upper():4s} {order.symbol:10s} "
            f"qty={int(order.qty):6d}  status={order.status}"
        )

    # Portfolio snapshot
    summary = broker.portfolio_summary()
    logger.info(f"\nPortfolio equity: ${summary['equity']:,.2f}")
    logger.info(f"Cash:             ${summary['cash']:,.2f}")
    logger.info(f"Positions:        {summary['num_positions']}")


# ---------------------------------------------------------------------------
# Command: watch（盯盤模式）
# ---------------------------------------------------------------------------

def _build_broker(config: dict):
    """
    Broker 工廠函數：根據 execution.broker 配置創建對應的 broker 實例。

    支持的類型：
      "futu_paper" — Futu 官方模擬環境（TrdEnv.SIMULATE）
      "paper"      — 內存模擬（無需 Futu，適合離線測試）
      "alpaca"     — Alpaca 美股（此處不實現，留佔位）
    """
    exec_cfg = config.get("execution", {})
    broker_type = exec_cfg.get("broker", "paper")
    risk_cfg = config.get("risk", {})

    if broker_type == "futu_paper":
        from src.execution.futu_trade_broker import FutuTradeBroker
        fp_cfg = exec_cfg.get("futu_paper", {})
        futu_conn = exec_cfg.get("futu", {})
        broker = FutuTradeBroker(
            host=futu_conn.get("host", "127.0.0.1"),
            port=futu_conn.get("port", 11111),
            market=fp_cfg.get("market", "hk"),
            acc_index=fp_cfg.get("acc_index", 0),
            lot_size=fp_cfg.get("lot_size", 100),
            slippage_bps=fp_cfg.get("slippage_bps", 30.0),
        )
        logger.info(
            f"Broker: FutuTradeBroker [SIMULATE]  "
            f"market={fp_cfg.get('market', 'hk')}  "
            f"host={futu_conn.get('host', '127.0.0.1')}:{futu_conn.get('port', 11111)}"
        )
    else:
        # 默認 PaperBroker（內存模擬）
        from src.execution.paper_broker import PaperBroker
        paper_cfg = exec_cfg.get("paper", {})
        broker = PaperBroker(
            initial_cash=paper_cfg.get("initial_cash", 1_000_000),
            slippage_bps=config.get("backtest", {}).get("slippage_bps", 5),
            commission=risk_cfg.get("commission_rate", 0.001),
        )
        logger.info("Broker: PaperBroker [in-memory simulation]")

    return broker


def cmd_watch(args: argparse.Namespace, config: dict) -> None:
    """
    持續盯盤模式：
    - 定時（收盤後）自動計算策略信號並推送通知
    - 每隔 N 秒檢查持倉止損 / 止盈 / 熔斷狀態
    - 智能風控：實時波動率/回撤/VaR + 市況狀態機 + 黑天鵝預警
    - (可選) 根據信號在 Futu 模擬盤下單

    啟動後阻塞運行，按 Ctrl-C 退出。
    """
    from src.notifier import Notifier
    from src.risk.manager import RiskManager
    from src.scheduler import WatchScheduler

    strategy_name = args.strategy or config.get("strategy", {}).get("active", "momentum_12_1")

    # 覆蓋 config 中的 strategy.active（讓 Scheduler 也感知到）
    config.setdefault("strategy", {})["active"] = strategy_name

    notify_cfg = config.get("notify", {})
    risk_cfg = config.get("risk", {})
    watch_cfg = config.get("watch", {})
    exec_cfg = config.get("execution", {})
    broker_type = exec_cfg.get("broker", "paper")

    notifier = Notifier(notify_cfg)
    risk_manager = RiskManager(risk_cfg)
    broker = _build_broker(config)

    logger.info("=" * 60)
    logger.info("  盯盤模式啟動（含智能風控）")
    logger.info(f"  策略    : {strategy_name}")
    logger.info(f"  Broker  : {broker_type}")
    logger.info(f"  信號Cron: {watch_cfg.get('signal_cron', '0 16 * * 1-5')}")
    logger.info(f"  下單模式: {'自動下單' if watch_cfg.get('trade_on_signal') else '僅通知'}")
    logger.info(f"  週報Cron: {watch_cfg.get('weekly_report_cron', '0 18 * * 5')}")
    logger.info(f"  通知渠道: {notify_cfg.get('channels', [])}")
    logger.info("=" * 60)

    # 啟動前發一條確認通知
    smart_cfg = risk_cfg.get("smart_risk", {})
    notifier.send_raw(
        subject="【Auto Trader】盯盤模式已啟動",
        body=(
            f"策略：{strategy_name}\n"
            f"Broker：{broker_type}\n"
            f"信號任務：{watch_cfg.get('signal_cron', '0 16 * * 1-5')}（Cron）\n"
            f"自動下單：{'是' if watch_cfg.get('trade_on_signal') else '否（僅通知）'}\n"
            f"週報任務：{watch_cfg.get('weekly_report_cron', '0 18 * * 5')}（Cron）\n"
            f"智能風控：{'已啟用' if smart_cfg.get('enabled', True) else '已禁用'}\n"
            f"  市況狀態機：{smart_cfg.get('regime_fast_ma', 20)}日快線 / {smart_cfg.get('regime_slow_ma', 60)}日慢線\n"
            f"  黑天鵝閾值：Z-score ≥ {smart_cfg.get('black_swan_zscore', 3.5)}σ\n"
            f"通知渠道：{notify_cfg.get('channels', [])}\n\n"
            f"啟動時間：{__import__('datetime').datetime.now()}\n"
            "— Auto Trader"
        ),
    )

    scheduler = WatchScheduler(
        config=config,
        notifier=notifier,
        broker=broker,
        risk_manager=risk_manager,
    )
    scheduler.start()  # 阻塞，直到 Ctrl-C


# ---------------------------------------------------------------------------
# Command: list strategies
# ---------------------------------------------------------------------------

def cmd_list(_args: argparse.Namespace, _config: dict) -> None:
    from src.strategies import list_strategies
    strategies = list_strategies()
    print("\n已注冊的策略：")
    for name in strategies:
        print(f"  • {name}")
    print()


# ---------------------------------------------------------------------------
# Command: info
# ---------------------------------------------------------------------------

def cmd_info(_args: argparse.Namespace, config: dict) -> None:
    import json
    print("\n當前生效配置：")
    print(json.dumps(config, indent=2, default=str, ensure_ascii=False))
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trader",
        description="自動選股交易系統",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="配置文件路徑（默認：config/config.yaml）"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # backtest
    bt = sub.add_parser("backtest", help="運行歷史回測")
    bt.add_argument("--strategy", help="策略名稱（覆蓋配置文件）")
    bt.add_argument("--start", help="開始日期 YYYY-MM-DD（覆蓋配置文件）")
    bt.add_argument("--end", help="結束日期 YYYY-MM-DD（覆蓋配置文件）")
    bt.add_argument("--no-plot", action="store_true", help="跳過結果圖表")
    bt.add_argument("--fast", action="store_true", help="限制 50 只股票（快速測試）")

    # paper
    pt = sub.add_parser("paper", help="運行一次紙面交易")
    pt.add_argument("--strategy", help="策略名稱（覆蓋配置文件）")
    pt.add_argument("--fast", action="store_true", help="限制 50 只股票")

    # watch（盯盤模式）
    wt = sub.add_parser("watch", help="持續盯盤：定時信號 + 止損監控 + 實時通知")
    wt.add_argument("--strategy", help="策略名稱（覆蓋配置文件）")

    # list
    sub.add_parser("list", help="列出所有已注冊的策略")

    # info
    sub.add_parser("info", help="打印當前生效配置")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    _setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    dispatch = {
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "watch": cmd_watch,
        "list": cmd_list,
        "info": cmd_info,
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()
