from __future__ import annotations
from abc import abstractmethod, ABCMeta

from ..types import LabeledHandPoints


class LabeledHandPointsStore(metaclass=ABCMeta):
    """
    学習データの保存と読み込みを受け持つインターフェース．

    add: 学習データを保存(追記)していく処理 (LabeledHandPoints) -> None
    load: 保存した学習データを読み込む処理 (void) -> list[LabeledHandPoints]

    """

    @abstractmethod
    def add(self, labeled_handpoints: LabeledHandPoints) -> None:
        ...

    @abstractmethod
    def load(self) -> list[LabeledHandPoints]:
        ...
