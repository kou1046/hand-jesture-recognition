from __future__ import annotations
import os

import torch
from torch import nn, optim
from torch.utils import data
import numpy as np
from sklearn.model_selection import train_test_split


from lib import LabeledHandPoints


NUM_CLASSES = 8


class HandSignDataset(data.Dataset):
    def __init__(self, dataset: list[LabeledHandPoints]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]

        return data.handpoints._to_relative()._normalize().to_numpy(), data.label - 1


class HandSignClassifier(nn.Module):
    def __init__(self, pretrained_model_path: None | str = None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.block2 = nn.Sequential(
            nn.Linear(2688, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NUM_CLASSES)
        )

        if pretrained_model_path:
            self.load_state_dict(
                torch.load(
                    pretrained_model_path,
                )["model_state_dict"]
            )
            self.eval()

    def forward(self, x):
        y = self.block(x)
        y = y.view(x.shape[0], -1)
        y = self.block2(y)
        return y

    def predict(self, handpoints):
        x = torch.Tensor([handpoints._to_relative()._normalize().to_numpy()])

        return int(torch.argmax(self(x), dim=1)[0])


def collate_fn(batch: list[tuple[np.ndarray, int]]):
    batch_points = []
    batch_labels = []
    for points, label in batch:
        batch_points.append(torch.Tensor(points))
        batch_labels.append(label)
    return torch.stack(batch_points), torch.Tensor(batch_labels).long()


if __name__ == "__main__":
    MAX_EPOCH = 100
    BATCH_SIZE = 100

    from lib import NdJsonLabeledHandPointsStore

    store = NdJsonLabeledHandPointsStore(os.path.join("models", "labeled_handpoints.ndjson"))
    labeled_handpoints = store.load()
    train, test = train_test_split(labeled_handpoints)

    train = HandSignDataset(train)
    test = HandSignDataset(test)

    train_loader = data.DataLoader(train, BATCH_SIZE, collate_fn=collate_fn)
    test_loader = data.DataLoader(test, BATCH_SIZE, collate_fn=collate_fn)

    model = HandSignClassifier()
    optim_ = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_accs = []
    test_accs = []

    checkpoints = [50, 70, 100]

    for epoch in range(1, MAX_EPOCH + 1):
        sum_acc = 0
        model.train()
        for points, t in train_loader:
            pred_y = model(points)
            loss = criterion(pred_y, t)
            model.zero_grad()
            loss.backward()
            optim_.step()
            sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
        train_acc = float(sum_acc / len(train_loader.dataset))
        print(f"train acc:{round(train_acc, 2)}")
        train_accs.append(train_acc)
        sum_acc = 0
        model.eval()
        with torch.no_grad():
            for points, t in train_loader:
                pred_y = model(points)
                sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
            test_acc = float(sum_acc / len(train_loader.dataset))
            print(f"test acc:{round(test_acc, 2)} epoch {epoch}/{MAX_EPOCH} done.")
            test_accs.append(test_acc)
            if epoch in checkpoints:
                ts = []
                preds_ys = []
                for points, t in train_loader:
                    ts += t.tolist()
                    preds_ys += torch.argmax(model(points), dim=1).tolist()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "train_accs": train_accs,
                        "test_accs": test_accs,
                        "test_pred_y": preds_ys,
                        "test_true_y": ts,
                    },
                    os.path.join("models", "hand_sign_classifier.pth"),
                )
