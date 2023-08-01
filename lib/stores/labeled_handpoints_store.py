from __future__ import annotations
from abc import abstractmethod, ABCMeta

from ..types import LabeledHandPoints


class LabeledHandPointsStore(metaclass=ABCMeta):
    @abstractmethod
    def save(self, labeled_handpoints: LabeledHandPoints) -> None:
        ...

    @abstractmethod
    def load(self) -> list[LabeledHandPoints]:
        ...
