"""
Import all strategy modules here so their @register_strategy decorators
fire at import time and populate the registry.

When you add a new strategy file, add its import here.
"""

from .base import AbstractStrategy
from .registry import register_strategy, get_strategy, list_strategies

# Built-in strategies — import to trigger registration
from . import momentum              # noqa: F401  registers "momentum_12_1"
from . import mean_reversion        # noqa: F401  registers "mean_reversion"
from . import multi_factor          # noqa: F401  registers "multi_factor"

# Lean-ported short-term strategies
from . import macd_strategy         # noqa: F401  registers "macd"
from . import ema_cross_strategy    # noqa: F401  registers "ema_cross"
from . import rsi_contrarian_strategy  # noqa: F401  registers "rsi_contrarian"

# A 股 / 港股 / 美股 穩健短線策略（7 條件全滿足）
from . import steady_short_term     # noqa: F401  registers "steady_short_term"

# 港股版別名（4 條件精簡版，通過 config 控制行為，與 CN 版同一實現）
from .registry import register_strategy as _reg
from .steady_short_term import SteadyShortTermStrategy as _SST
_reg("steady_short_term_hk")(_SST)  # noqa: registers "steady_short_term_hk"

# 美股趨勢策略（60 日均線 + 支撐入場，ETF + 龍頭科技固定池）
from . import us_trend_strategy     # noqa: F401  registers "us_trend"

__all__ = [
    "AbstractStrategy",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "SteadyShortTermStrategy",
    "USTrendStrategy",
]
