from __future__ import annotations

import torch
from torch import nn

from ..types import HandPoints


class HandSignClassifier(nn.Module):
    def __init__(self, output_size, pretrained_model_path: None | str = None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.block2 = nn.Sequential(
            nn.Linear(2688, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_size)
        )

        if pretrained_model_path:
            self.load_state_dict(
                torch.load(
                    pretrained_model_path,
                )["model_state_dict"]
            )
            self.eval()

    def forward(self, x: torch.Tensor):
        y = self.block(x)
        y = y.view(x.shape[0], -1)
        y = self.block2(y)
        return y

    def predict(self, handpoints: HandPoints):
        """
        実際の運用時はこっちを用いるとよい. HandPointsを渡すと予測ラベルを返す.
        """
        x = torch.Tensor([handpoints.preprocess()])

        return int(torch.argmax(self(x), dim=1)[0])
