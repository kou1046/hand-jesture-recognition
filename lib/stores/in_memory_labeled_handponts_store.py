from __future__ import annotations

from .labeled_handpoints_store import LabeledHandPointsStore, LabeledHandPoints
from ..types import LabeledHandPoints


class InMemoryLabeledHandpointsStore(LabeledHandPointsStore):
    """
    メモリ内にデータを保持するストア．永続化はしない．主にテスト用や学習データを保存する必要がないときに使う．
    """

    def __init__(self):
        self.store: list[LabeledHandPoints] = []

    def add(self, labeled_handpoints: LabeledHandPoints) -> None:
        self.store.append(labeled_handpoints)

    def load(self) -> list[LabeledHandPoints]:
        return self.store.copy()
