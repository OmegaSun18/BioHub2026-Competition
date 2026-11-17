"""Tiny name -> object registries. A new feature builder, model or pipeline stage is one decorated class; configs refer to it by name."""
from __future__ import annotations


class Registry:
    def __init__(self, kind: str):
        self.kind, self._items = kind, {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            obj.registry_name = name
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)

    def __contains__(self, name):
        return name in self._items


FEATURES = Registry("feature builder")
MODELS = Registry("model")
STAGES = Registry("pipeline stage")
