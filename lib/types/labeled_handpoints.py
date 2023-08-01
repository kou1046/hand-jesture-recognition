from dataclasses import dataclass

from .handpoints import HandPoints


@dataclass(frozen=True)
class LabeledHandPoints:
    label: int
    handpoints: HandPoints
