"""
Strategy registry: a simple plugin system for strategies.

Usage
-----
# Registering (in your strategy file):
from src.strategies.registry import register_strategy

@register_strategy("my_paper_strategy")
class MyPaperStrategy(AbstractStrategy):
    ...

# Looking up at runtime:
from src.strategies.registry import get_strategy

StrategyClass = get_strategy("my_paper_strategy")
strategy = StrategyClass(config={"top_n": 20})

# Listing all registered strategies:
from src.strategies.registry import list_strategies
print(list_strategies())
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import AbstractStrategy

_REGISTRY: Dict[str, Type[AbstractStrategy]] = {}


def register_strategy(name: str):
    """
    Class decorator that registers a strategy under the given name.

    Parameters
    ----------
    name : unique string key, used in config.yaml ``strategy.active``
    """
    def decorator(cls: Type[AbstractStrategy]) -> Type[AbstractStrategy]:
        if name in _REGISTRY:
            raise ValueError(
                f"Strategy '{name}' is already registered. "
                "Choose a different name or remove the duplicate."
            )
        if not issubclass(cls, AbstractStrategy):
            raise TypeError(
                f"{cls.__name__} must inherit from AbstractStrategy"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str) -> Type[AbstractStrategy]:
    """
    Return the strategy class registered under ``name``.

    Raises KeyError with a helpful message listing available strategies.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none registered yet)"
        raise KeyError(
            f"Unknown strategy '{name}'. Available strategies: {available}"
        )
    return _REGISTRY[name]


def list_strategies() -> List[str]:
    """Return all registered strategy names in alphabetical order."""
    return sorted(_REGISTRY.keys())
