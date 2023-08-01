from __future__ import annotations
from dataclasses import asdict

import ndjson
from dacite import from_dict

from .labeled_handpoints_store import LabeledHandPointsStore
from ..types import LabeledHandPoints


class NdJsonLabeledHandPointsStore(LabeledHandPointsStore):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, labeled_handpoints: LabeledHandPoints) -> None:
        with open(self.file_path, "a") as f:
            writer = ndjson.writer(f)
            writer.writerow(asdict(labeled_handpoints))

    def load(self) -> list[LabeledHandPoints]:
        with open(self.file_path, "r") as f:
            dataset: list[dict] = ndjson.load(f)
            dataset = [from_dict(LabeledHandPoints, data) for data in dataset]
        return dataset
