# 自動選股交易系統

一個插件化、多市場的自動選股與交易系統，內置向量化回測引擎、紙面/實盤交易執行層，以及自動盯盤 + 郵件通知功能。

---

## 功能特性

| 功能 | 說明 |
|---|---|
| **多市場** | 美股（S&P 500，via yfinance）、港股 & A 股（via akshare） |
| **內置策略** | 動量、均值回歸、多因子 + **A股穩健短線**（`steady_short_term`）+ **港股穩健短線**（`steady_short_term_hk`）+ **美股趨勢**（`us_trend`，MA60 + 支撐入場）——均可插件式替換 |
| **回測引擎** | 向量化引擎，模擬真實滑點與手續費 |
| **執行層** | 紙面交易（內存模擬）/ **Futu 模擬盤（TrdEnv.SIMULATE）** / Alpaca（可選） |
| **智能風控** | 實時風險指標（波動率/回撤/VaR/集中度）+ 市況狀態機（牛/中性/熊）+ 黑天鵝預警（觸發緊急縮倉/清倉） |
| **自動盯盤** | 收盤後信號推送 + （可選）信號自動下單 + 盤中止損/止盈/黑天鵝實時警報 |
| **收盤自動復盤** | 每日收盤自動發郵件：逐筆交易「為什麼賺/虧」+ 收益歸因（趨勢/震盪/行情/殘差）+ 改進建議（不必看數據） |
| **策略自動升級（建議）** | 根據最近表現自動產出調參/淘汰/新規律建議（僅輸出建議，不自動改策略代碼） |
| **通知渠道** | 郵件（SMTP）、企業微信 Webhook（可選）|
| **可擴展** | 約 30 行代碼即可接入任意論文策略，詳見 `STRATEGY_GUIDE.md` |

---

## 快速開始

### 第一步：安裝依賴

```bash
cd /path/to/trader

# 創建並激活虛擬環境
python3 -m venv .venv
source .venv/bin/activate      # Windows 用戶: .venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 第二步：配置郵件通知

在 `config/config.yaml` 的 `notify.email` 節填入郵箱信息：

```yaml
notify:
  channels:
    - email               # 只啟用郵件

  email:
    smtp_host: "smtp.qq.com"   # Gmail 用 smtp.gmail.com
    smtp_port: 587
    use_tls: true
    username: "your@qq.com"
    password: ""               # 留空，密碼通過環境變量傳入（見下方）
    from_addr: "your@qq.com"
    to_addrs:
      - "your@qq.com"
```

**密碼通過環境變量傳入**（避免明文寫入文件）：

```bash
# 寫入 ~/.zshrc，永久生效
echo 'export EMAIL_PASSWORD="你的郵箱密碼"' >> ~/.zshrc
source ~/.zshrc
```

> QQ 郵箱需使用「授權碼」而非登錄密碼：  
> QQ 郵箱網頁版 → 設置 → 賬戶 → 開啟 SMTP → 生成授權碼

### 第三步：配置自選股盯盤名單

在 `config/config.yaml` 的 `watch.watchlist` 填入股票代碼：

```yaml
watch:
  watchlist:
    - "00700"   # 騰訊控股（港股）
    - "AAPL"    # 蘋果（美股）
    # - "600519"  # 貴州茅台（A 股）
```

### 第四步：啟動盯盤服務

```bash
# 前台運行（Ctrl-C 停止）
python3 main.py watch

# 後台持久運行（關閉終端後繼續）
mkdir -p logs
nohup python3 main.py watch > logs/watch.log 2>&1 &

# 查看實時日誌
tail -f logs/watch.log

# 停止後台服務
kill $(pgrep -f "main.py watch")
```

啟動成功後，你的郵箱會收到一封確認郵件：**「Auto Trader 盯盤模式已啟動」**。

---

## Futu 模擬盤（TrdEnv.SIMULATE）

如需「信號→自動下單」以及更接近真實的模擬環境，請使用 **FutuOpenD 模擬盤**：

1. 安裝依賴：

```bash
pip install futu-api
```

2. 啟動本地 FutuOpenD（默認 `127.0.0.1:11111`）

3. 配置 `config/config.yaml`：

```yaml
execution:
  broker: "futu_paper"        # ✅ 使用 Futu 模擬盤
  futu:
    host: "127.0.0.1"
    port: 11111
  futu_paper:
    market: "hk"              # hk | us | multi
    acc_index: 0

watch:
  trade_on_signal: true       # ✅ 收盤後信號自動下單
  weekly_report_cron: "0 18 * * 5"
```

> 重要：本項目所有 Futu 下單均固定使用 `TrdEnv.SIMULATE`（不會下到真實賬戶）。

---

## 其他命令

### 運行歷史回測

```bash
# 默認策略（momentum_12_1）在 S&P 500 上回測（2018-2024）
python3 main.py backtest

# 快速冒煙測試：限制 50 只股票（約 1-2 分鐘）
python3 main.py backtest --fast

# 指定策略和時間範圍
python3 main.py backtest --strategy multi_factor --start 2020-01-01 --end 2023-12-31
```

### 策略使用示例（A 股 / 港股 / 美股）

> 策略均可用 `python3 main.py list` 查看是否已注冊。

**A 股｜穩健短線（7 條件全滿足）**：`steady_short_term`

```yaml
market:
  mode: "cn"
strategy:
  active: "steady_short_term"
```

**港股｜穩健短線（4 條件精簡版）**：`steady_short_term_hk`

```yaml
market:
  mode: "hk"
strategy:
  active: "steady_short_term_hk"
```

**美股｜趨勢策略（MA60 + 支撐入場）**：`us_trend`

```yaml
market:
  mode: "us"
strategy:
  active: "us_trend"
```

### 運行一次紙面交易

```bash
# 拉取今日數據，生成信號，模擬下單
python3 main.py paper

# 指定策略
python3 main.py paper --strategy momentum_12_1
```

### 其他工具命令

```bash
python3 main.py list    # 列出所有已注冊策略
python3 main.py info    # 打印當前生效配置
```

---

## 系統架構

```
trader/
├── main.py                     CLI 入口
├── config/
│   └── config.yaml             所有配置（市場、策略、風控、Broker）
└── src/
    ├── data/                   市場數據層
    │   ├── base_fetcher.py     抽象接口 + 磁盤緩存
    │   ├── us_fetcher.py       yfinance（美股）
    │   ├── cn_hk_fetcher.py    akshare（港股 / A 股）
    │   └── preprocessor.py    數據清洗、收益率、動量、Z-score、RSI
    ├── strategies/             策略插件系統
    │   ├── base.py             AbstractStrategy + @register_strategy
    │   ├── registry.py         策略注冊表
    │   ├── momentum.py         12-1 動量（Jegadeesh & Titman 1993）
    │   ├── mean_reversion.py   短期均值回歸（De Bondt & Thaler 1985）
    │   ├── multi_factor.py     動量 + 低波動 + 質量 合成因子
    │   ├── steady_short_term.py A股穩健短線（7 條件全滿足）
    │   ├── us_trend_strategy.py 美股趨勢（MA60 + 支撐入場，ETF+龍頭科技固定池）
    │   └── ...                 其它策略（Lean 移植 / 自定義策略）
    ├── selection/
    │   ├── screener.py         通用篩選器（價格/市值/流動性/上市時間）
    │   ├── cn_screener.py      A股前置過濾（ST/停牌/市值/換手率/暴跌&跌停）
    │   └── hk_screener.py      港股前置過濾（股價/推算流通市值/換手率/負資產/停牌）
    ├── backtest/
    │   ├── engine.py           向量化再平衡回測引擎
    │   └── metrics.py          Sharpe、Sortino、最大回撤、Alpha/Beta
    ├── execution/
    │   ├── base_broker.py      抽象 Broker 接口
    │   ├── paper_broker.py     內存模擬交易
    │   ├── futu_trade_broker.py Futu 模擬盤（TrdEnv.SIMULATE）
    │   └── alpaca_broker.py    Alpaca REST API（實盤，可選）
    ├── analysis/
    │   ├── trade_reviewer.py   逐筆交易復盤（信號/執行/行情）
    │   ├── attribution.py      收益歸因（趨勢/震盪/Alpha/殘差）
    │   ├── strategy_optimizer.py 策略升級建議（僅輸出建議）
    │   └── report_generator.py 日報/周報（結論先行）
    └── risk/
        └── manager.py          倉位管理、止損追蹤、熔斷機制
```

### 數據流

```
股票池 → 數據拉取 → 數據清洗 → 股票篩選 → 策略信號 → 風控定倉
                                                           ↓
                                          回測引擎 / 紙面交易 Broker
                                                           ↓
                                              績效報告 / 訂單執行
```

---

## 配置說明

編輯 `config/config.yaml` 調整所有行為，核心配置項：

```yaml
market:
  mode: "us"                  # us | hk | cn | multi

data:
  start_date: "2018-01-01"
  end_date:   "2024-12-31"

strategy:
  active: "momentum_12_1"     # 啟用的策略名稱
  momentum_12_1:
    lookback_months: 12       # 動量回望窗口（月）
    skip_months:     1        # 跳過最近 N 個月（避免短期反轉）
    top_n:           20       # 持倉股票數量
    rebalance_freq:  "M"      # 再平衡頻率：M 月 | W 週 | Q 季

risk:
  max_position_weight: 0.10   # 單股最大倉位 10%
  stop_loss_pct:       0.08   # 虧損 8% 觸發止損
  max_drawdown_halt:   0.20   # 組合回撤 20% 熔斷
  smart_risk:
    enabled: true             # ✅ 智能風控總開關
    exposure:
      bull: 0.40              # 牛市最大倉位
      neutral: 0.20
      bear: 0.05
    black_swan_emergency_exposure: 0.0  # 黑天鵝觸發後目標倉位（0=清倉）

execution:
  broker: "futu_paper"        # paper（內存模擬）| futu_paper（Futu 模擬盤）| alpaca（可選）

watch:
  trade_on_signal: true        # ✅ 收盤信號是否自動下單
  weekly_report_cron: "0 18 * * 5"
```

### 新增策略與前置過濾器（本倉庫擴展）

- **A 股前置過濾**：`src/selection/cn_screener.py`（ST/停牌/股價/流通市值/換手率 + 近 1 個月暴跌/連續跌停排除）
- **港股前置過濾**：`src/selection/hk_screener.py`（股價 ≥ 1、換手率 0.5%-8%、推算流通市值 50 億-2000 億、排除負資產/停牌）
  - 港股快照 `ak.stock_hk_spot_em()` 無直接市值欄位，使用公式推算：\n    \(\n    \\text{float\\_cap} = \\frac{\\text{volume}}{\\text{turnover}/100} \\times \\text{price}\n    \)\n+- **A 股穩健短線**：`steady_short_term`（MA5>MA10>MA20 + 站線上 + 溫和放量 + 形態/連漲限制 + 1%-4%）\n+- **港股穩健短線**：`steady_short_term_hk`（同一實現的配置別名，關閉形態/連漲限制，漲幅 2%-6%）\n+- **美股趨勢策略**：`us_trend`（固定 ETF+龍頭科技池 + MA60 趨勢 + 近支撐入場，不玩短期金叉死叉）\n+
完整配置以 `config/config.yaml` 內注釋為準。

---

## 添加論文策略（3 步接入）

詳細指南見 **`STRATEGY_GUIDE.md`**，核心流程：

**第一步**：新建 `src/strategies/my_strategy.py`

```python
from src.strategies.base import AbstractStrategy
from src.strategies.registry import register_strategy
import pandas as pd

@register_strategy("my_strategy")          # 唯一策略名稱
class MyStrategy(AbstractStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 基於 data（全量歷史 OHLCV）計算信號
        # 返回 pd.Series：+1 買入，-1 賣出，0 不操作
        ...

    def get_params(self) -> dict:
        return {"strategy": "my_strategy"}
```

**第二步**：在 `src/strategies/__init__.py` 添加一行 import：

```python
from . import my_strategy   # noqa: F401
```

**第三步**：在 `config/config.yaml` 啟用並配置參數，然後運行：

```bash
python3 main.py backtest --strategy my_strategy
```

---

## 實盤交易（Alpaca）

1. 安裝可選依賴：`pip install alpaca-trade-api`
2. 設置環境變量：
   ```bash
   export ALPACA_API_KEY=你的Key
   export ALPACA_SECRET_KEY=你的Secret
   export ALPACA_BASE_URL=https://paper-api.alpaca.markets   # 沙盒環境
   ```
3. 在 `config/config.yaml` 中設置 `execution.broker: "alpaca"`
4. 運行：`python3 main.py paper`

> **注意**：請務必先在 Alpaca 沙盒環境（paper trading）充分測試後，
> 再切換至實盤接口。作者對任何交易損失不承擔責任。

---

## 免責聲明

本軟件僅供學習和研究使用，不構成任何投資建議。
任何策略的歷史回測表現不代表未來收益。所有交易均存在虧損風險。
