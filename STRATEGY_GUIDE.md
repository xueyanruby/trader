# 論文策略接入手冊

如何將研究論文中的策略接入本系統——完整的五步指南。

---

## 目錄

1. [系統接口約定](#1-系統接口約定)
2. [從論文到代碼：五步流程](#2-從論文到代碼五步流程)
3. [完整示例：低波動率異象](#3-完整示例低波動率異象)
4. [常用因子實現速查](#4-常用因子實現速查)
5. [多因子組合](#5-多因子組合)
6. [運行與評估策略](#6-運行與評估策略)
7. [常見陷阱與規避方法](#7-常見陷阱與規避方法)
8. [進階主題](#8-進階主題)

---

## 1. 系統接口約定

每個策略必須實現兩個方法：

```python
def generate_signals(self, data: pd.DataFrame) -> pd.Series:
    """
    data  : MultiIndex DataFrame
            列索引 = (symbol, field)，行索引 = DatetimeIndex
            可用 field：open, high, low, close, volume
            包含截止到當前再平衡日的【全量歷史數據】
            ❗ 嚴禁使用最後一行之後的數據，否則造成前視偏差

    返回值: pd.Series
            index = 股票代碼
            value ∈ {+1, 0, -1}
            +1  → 做多（買入）
            -1  → 做空（當前引擎暫不使用）
             0  → 不持倉（可以從 Series 中省略）
    """

def get_params(self) -> dict:
    """返回策略參數的扁平字典，用於回測記錄和可復現性"""
```

引擎在每個再平衡日調用 `generate_signals()`，傳入截止該日的全量歷史數據。
風控模塊隨後將信號轉換為帶權重的目標倉位。

---

## 2. 從論文到代碼：五步流程

### 第一步 — 閱讀論文，找出核心公式

重點關注：
- **信號/因子公式**（每只股票計算什麼）
- **排序方法**（Top N、十分位、閾值）
- **再平衡頻率**（月度、週度、季度）
- **估計窗口**（需要多少個月/天的歷史數據）

### 第二步 — 將公式映射到可用數據字段

| 論文中的概念 | 系統中的對應方法 |
|---|---|
| 收盤價 | `Preprocessor.get_close(data)` 或 `data[symbol]["close"]` |
| 日收益率 | `Preprocessor.compute_returns(data)` |
| 成交量 | `Preprocessor.get_volume(data)` 或 `data[symbol]["volume"]` |
| 歷史波動率 | `Preprocessor.compute_volatility(data)` |
| 動量因子 | `Preprocessor.compute_momentum(data)` |
| RSI | `Preprocessor.compute_rsi(data)` |
| Z-score | `Preprocessor.compute_z_score(data)` |
| 市盈率/市淨率等基本面 | 需先調用 `fetcher.get_info(symbols)` 單獨獲取 |

### 第三步 — 創建策略文件

```bash
touch src/strategies/my_paper.py
```

代碼模板：

```python
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy

@register_strategy("my_paper_strategy")   # ← 唯一策略名稱，與 config.yaml 對應
class MyPaperStrategy(AbstractStrategy):

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        top_n: int = self._cfg("top_n", 20)

        # ===== 在此處實現論文因子計算 =====
        scores = ...      # pd.Series：index=股票代碼，value=因子得分

        # 選出得分最高的 Top N 只股票
        top = scores.nlargest(top_n).index
        signals = pd.Series(0, index=scores.index, dtype=int)
        signals[top] = 1
        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "my_paper_strategy",
            "top_n": self._cfg("top_n", 20),
        }
```

### 第四步 — 注冊策略

在 `src/strategies/__init__.py` 中添加一行 import，觸發裝飾器注冊：

```python
from . import my_paper      # noqa: F401  注冊 "my_paper_strategy"
```

### 第五步 — 配置並運行

在 `config/config.yaml` 中添加配置：

```yaml
strategy:
  active: "my_paper_strategy"
  my_paper_strategy:
    top_n: 20
    rebalance_freq: "M"
    # 其他策略所需參數
```

運行回測：

```bash
python3 main.py backtest --strategy my_paper_strategy --fast
```

---

## 3. 完整示例：低波動率異象

**論文來源**：Blitz & van Vliet（2007）
《The Volatility Effect: Lower Risk without Lower Return》
*Journal of Portfolio Management*

**核心觀點**：歷史波動率越低的股票，風險調整後收益越好——這與 CAPM 的預測相反。

**信號公式**：按 36 個月歷史波動率排序，買入波動率最低的 N 只股票。

```python
# src/strategies/low_volatility.py

from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..data.preprocessor import Preprocessor
from .base import AbstractStrategy
from .registry import register_strategy

@register_strategy("low_volatility")
class LowVolatilityStrategy(AbstractStrategy):
    """
    低波動率異象策略（Blitz & van Vliet 2007）

    買入歷史波動率最低的 N 只股票，月度再平衡。

    配置項
    ------
    lookback_days  : 估計窗口（天），默認 756 ≈ 36 個月
    top_n          : 持倉數量，默認 20
    rebalance_freq : 再平衡頻率，默認 "M"
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        lookback_days: int = self._cfg("lookback_days", 756)
        top_n: int = self._cfg("top_n", 20)

        # 計算各股票 36 個月年化波動率
        vol = Preprocessor.compute_volatility(data, lookback_days=lookback_days)

        if vol.empty:
            return pd.Series(dtype=float)

        # 買入波動率最低的 N 只股票
        bottom = vol.nsmallest(top_n).index
        signals = pd.Series(0, index=vol.index, dtype=int)
        signals[bottom] = 1
        return signals[signals != 0]

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": "low_volatility",
            "lookback_days": self._cfg("lookback_days", 756),
            "top_n": self._cfg("top_n", 20),
            "rebalance_freq": self._cfg("rebalance_freq", "M"),
            "paper": "Blitz & van Vliet (2007)",
        }
```

在 `src/strategies/__init__.py` 添加：
```python
from . import low_volatility   # noqa: F401
```

運行：
```bash
python3 main.py backtest --strategy low_volatility
```

---

## 4. 常用因子實現速查

### 動量因子（12-1 月）

```python
# 來源：Jegadeesh & Titman (1993)
momentum = Preprocessor.compute_momentum(data, lookback_days=252, skip_days=21)
top = momentum.nlargest(top_n).index   # 買入動量最強的 N 只
```

### 短期反轉（1 週）

```python
# 來源：Lehmann (1990) — 買入近期跌幅最大的股票
close = Preprocessor.get_close(data)
short_ret = close.pct_change().iloc[-5:].sum()  # 最近 5 個交易日收益
bottom = short_ret.nsmallest(top_n).index       # 跌得最多 = 買入
```

### 價值因子（市淨率倒數）

```python
# 來源：Fama & French (1993)
# 需要基本面數據，單獨通過 get_info() 獲取
info = fetcher.get_info(symbols)
pb = info["price_to_book"].dropna()
bm = 1 / pb                                     # 賬面市值比 = 1 / 市淨率
top = bm.nlargest(top_n).index                  # BM 越高 = 價值股
```

### 盈利因子（盈利率）

```python
# 來源：Novy-Marx (2013) — 總盈利能力
info = fetcher.get_info(symbols)
pe = info["trailing_pe"].dropna()
ey = 1 / pe                                     # 盈利率 = 1 / 市盈率
top = ey.nlargest(top_n).index
```

### 低 Beta 因子（押注逆 Beta）

```python
# 來源：Frazzini & Pedersen (2014) — BAB 因子
import numpy as np

close = Preprocessor.get_close(data)
returns = close.pct_change().dropna()
bench_ret = returns[benchmark_symbol]           # 基準收益率

betas = {}
window = 252
for sym in returns.columns:
    if sym == benchmark_symbol:
        continue
    stock_r = returns[sym].iloc[-window:].dropna()
    bench_r = bench_ret.iloc[-window:].reindex(stock_r.index).dropna()
    if len(stock_r) < 60:
        continue
    cov = np.cov(stock_r, bench_r)[0, 1]
    var = np.var(bench_r)
    betas[sym] = cov / var if var > 0 else np.nan

beta_series = pd.Series(betas).dropna()
low_beta = beta_series.nsmallest(top_n).index  # 低 Beta 股票 = 買入
```

### 52 週高點動量

```python
# 來源：George & Hwang (2004)
close = Preprocessor.get_close(data)
high_52w = close.iloc[-252:].max()
last_price = close.iloc[-1]
nearness = last_price / high_52w               # 1.0 = 處於 52 週最高點附近
top = nearness.nlargest(top_n).index
```

---

## 5. 多因子組合

`multi_factor.py` 已示範基本模式。以下展示更通用的組合方式：

### 等權 Z-score 合成

```python
import numpy as np
import pandas as pd

def _zscore(s: pd.Series) -> pd.Series:
    """截面 Z-score 標準化"""
    return (s - s.mean()) / s.std()

# 計算各因子
f_mom     = _zscore(Preprocessor.compute_momentum(data))
f_low_vol = _zscore(-Preprocessor.compute_volatility(data))  # 取反：低波動 = 高分
f_value   = _zscore(value_factor)   # 來自基本面數據

# 加權合成（可配置）
composite = 0.4 * f_mom + 0.3 * f_low_vol + 0.3 * f_value

# 選出 Top N
top = composite.nlargest(top_n).index
```

### IC 加權組合（進階）

信息係數（IC）= 因子得分與下期收益的秩相關係數。
使用歷史 IC 均值作為各因子的組合權重，可以讓表現好的因子獲得更大比重。

```python
from scipy.stats import spearmanr

ic_values = []
for date in historical_rebalance_dates:
    scores = compute_factor_at(date)      # 該日因子得分
    fwd_ret = returns_next_period(date)   # 下一期實際收益
    ic, _ = spearmanr(scores, fwd_ret, nan_policy="omit")
    ic_values.append(ic)

ic_mean = np.mean(ic_values)   # 作為該因子在組合中的權重
```

---

## 6. 運行與評估策略

### 運行回測

```bash
python3 main.py backtest --strategy my_strategy
```

輸出指標包括：
- 總收益率、年化收益率、年化波動率
- 夏普比率（Sharpe Ratio）
- 索提諾比率（Sortino Ratio）
- 最大回撤（Max Drawdown）
- 卡瑪比率（Calmar Ratio）
- Alpha、Beta（相對基準）
- 信息比率（Information Ratio）
- 結果圖表保存至 `backtest_result.png`

### 參數敏感性分析

```python
# 在 Jupyter Notebook 中運行
from src.backtest.engine import BacktestEngine
import copy

results = []
for top_n in [10, 20, 30, 50]:
    cfg = copy.deepcopy(config)
    cfg["strategy"]["my_strategy"]["top_n"] = top_n
    engine = BacktestEngine(cfg)
    r = engine.run(prices, "my_strategy")
    rep = r.report()
    results.append({
        "top_n": top_n,
        "sharpe": rep["sharpe_ratio"],
        "max_dd": rep["max_drawdown"],
    })

import pandas as pd
pd.DataFrame(results)
```

### 樣本外驗證（Walk-Forward）

先用 2014-2019 的數據優化參數，再用 2020-2024 驗證：

```bash
# 樣本內：確定最優參數
python3 main.py backtest --strategy my_strategy --start 2014-01-01 --end 2019-12-31

# 樣本外：用固定參數評估真實表現
python3 main.py backtest --strategy my_strategy --start 2020-01-01 --end 2024-12-31
```

---

## 7. 常見陷阱與規避方法

### 陷阱 1：前視偏差（Look-Ahead Bias）

❌ **錯誤做法**：在 `generate_signals()` 內部發起額外的數據請求，或使用未來數據。

✅ **正確做法**：`generate_signals()` 的 `data` 參數已包含截止再平衡日的全量歷史。
只使用 `data` 中的數據，不在方法內部調用任何網絡請求。

```python
# ✅ 正確：只使用傳入的 data
close = Preprocessor.get_close(data)
momentum = close.pct_change(252).iloc[-1]

# ❌ 錯誤：在 generate_signals 裡另外拉取數據（可能包含未來信息）
# latest = fetcher.get_prices(symbols, start="2024-01-01", end="2024-12-31")
```

### 陷阱 2：倖存者偏差（Survivorship Bias）

yfinance 的 S&P 500 名單只包含**當前**成分股，不包含歷史上被剔除的公司（退市、破產等）。

**影響**：策略歷史表現被高估約 1-2% / 年。

**規避方法**：
- 在論文中明確標注倖存者偏差的存在
- 對報告收益做保守折扣
- 如條件允許，使用包含歷史退市股票的 CRSP 或 Bloomberg 數據

### 陷阱 3：交易成本低估

默認配置：手續費 0.1% + 滑點 5bps。
月度再平衡 20 只股票，年換手率約 100-200%，每年成本約 1.5-3%。

**規避方法**：回測後查看 `trade_log` 核實換手率，必要時調高 `commission_rate` 參數。

### 陷阱 4：數據窺探 / 過擬合

- 不要在同一樣本上既做參數選擇又報告績效。
- 使用 Walk-Forward 測試或嚴格的樣本外驗證。
- 策略參數超過 3 個時，需格外警惕過擬合風險。

### 陷阱 5：再平衡時機偏差

所有再平衡均在每期末最後一個交易日的收盤價執行。
**倉位建立在下一個交易日**（通過 `skip_days` 實現），避免使用當天收盤價計算並立即用同一收盤價成交的循環偏差。

### 陷阱 6：多市場貨幣不一致

同時跑美股 + 港股策略時，確保所有收益率都轉換為同一貨幣，或在合併信號前進行匯率調整。

---

## 8. 進階主題

### 接入機器學習信號

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@register_strategy("ml_random_forest")
class MLRandomForestStrategy(AbstractStrategy):
    """
    使用隨機森林預測下期收益方向。

    ⚠️ 注意：以下為示意代碼，簡化了標籤的生成方式。
    生產環境必須使用嚴格的時序交叉驗證，
    確保訓練集標籤不包含任何未來信息。
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        features = Preprocessor.compute_all_features(data)
        if len(features) < 50:
            return pd.Series(dtype=float)

        close = Preprocessor.get_close(data)
        # 用過去一個月的收益作為代理標籤（僅供演示，非生產代碼）
        past_ret = close.pct_change(21).iloc[-1]
        labels = (past_ret > 0).astype(int)

        common = features.index.intersection(labels.index)
        X = features.loc[common].values
        y = labels.loc[common].values

        if len(X) < 30:
            return pd.Series(dtype=float)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(features.values)[:, 1]
        scores = pd.Series(proba, index=features.index)

        top = scores.nlargest(self._cfg("top_n", 20)).index
        signals = pd.Series(0, index=scores.index)
        signals[top] = 1
        return signals[signals != 0]

    def get_params(self) -> dict:
        return {"strategy": "ml_random_forest", "top_n": self._cfg("top_n", 20)}
```

### 接入另類數據

任何外部數據源（新聞情緒、期權流量、盈利驚喜等）均可通過以下方式接入：
1. 實現一個繼承 `BaseFetcher` 的新子類
2. 在 `generate_signals()` 調用前獲取數據
3. 將數據合並到 `data` DataFrame 或通過 `self.config` 傳入

### 多市場信號合並

```python
@register_strategy("global_momentum")
class GlobalMomentumStrategy(AbstractStrategy):
    """跨市場動量：分別計算各市場的截面動量，再合並信號"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 分別為美股和港股計算動量，各取前十分位
        # 合並後形成全球組合
        us_signals  = self._compute_for_market(data, market="us")
        hk_signals  = self._compute_for_market(data, market="hk")
        return pd.concat([us_signals, hk_signals])

    def _compute_for_market(self, data, market):
        # 根據股票代碼前綴過濾對應市場的數據，計算動量因子
        ...

    def get_params(self) -> dict:
        return {"strategy": "global_momentum"}
```

### 期望論文接入流程總結

```
閱讀論文
  ↓
識別核心公式（因子 = f(price, volume, fundamentals)）
  ↓
確認數據可獲取性（OHLCV vs 基本面）
  ↓
在 Notebook 中驗證因子 IC（信息係數）
  ↓
實現 AbstractStrategy 子類
  ↓
注冊 + 添加配置
  ↓
回測（樣本內）
  ↓
樣本外驗證
  ↓
（可選）紙面交易實盤驗證
```
