from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils import data
import numpy as np

from sklearn.model_selection import train_test_split

from ..types import LabeledHandPoints
from .hand_sign_classifier import HandSignClassifier
from ..stores import LabeledHandPointsStore


class HandSignTrainer:
    CHECK_EPOCHS = [10, 30, 50]

    def __init__(self, store: LabeledHandPointsStore, classifier: HandSignClassifier):
        self.store = store
        self.classifier = classifier

    def train(self, model_save_path: str, batch_size: int = 128, max_epoch: int = 100):
        labeled_handpoints = self.store.load()
        train, test = train_test_split(labeled_handpoints)

        train = HandSignDataset(train)
        test = HandSignDataset(test)

        train_loader = data.DataLoader(train, batch_size, collate_fn=collate_fn)
        test_loader = data.DataLoader(test, batch_size, collate_fn=collate_fn)

        optim_ = optim.Adam(self.classifier.parameters())
        criterion = nn.CrossEntropyLoss()

        train_accs = []
        test_accs = []

        for epoch in range(1, max_epoch + 1):
            sum_acc = 0
            self.classifier.train()
            for points, t in train_loader:
                pred_y = self.classifier(points)
                loss = criterion(pred_y, t)
                self.classifier.zero_grad()
                loss.backward()
                optim_.step()
                sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
            train_acc = float(sum_acc / len(train_loader.dataset))
            print(f"train acc:{round(train_acc, 2)}")
            train_accs.append(train_acc)
            sum_acc = 0
            self.classifier.eval()
            with torch.no_grad():
                for points, t in test_loader:
                    pred_y = self.classifier(points)
                    sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
                test_acc = float(sum_acc / len(test_loader.dataset))
                print(f"test acc:{round(test_acc, 2)} epoch {epoch}/{max_epoch} done.")
                test_accs.append(test_acc)
                if epoch in self.CHECK_EPOCHS or epoch == max_epoch:
                    ts = []
                    preds_ys = []
                    for points, t in test_loader:
                        ts += t.tolist()
                        preds_ys += torch.argmax(self.classifier(points), dim=1).tolist()
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.classifier.state_dict(),
                            "train_accs": train_accs,
                            "test_accs": test_accs,
                            "test_pred_y": preds_ys,
                            "test_true_y": ts,
                        },
                        model_save_path,
                    )


class HandSignDataset(data.Dataset):
    def __init__(self, dataset: list[LabeledHandPoints]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset[index]

        return data.handpoints._to_relative()._normalize().to_numpy(), data.label - 1


def collate_fn(batch: list[tuple[np.ndarray, int]]):
    batch_points = []
    batch_labels = []
    for points, label in batch:
        batch_points.append(torch.Tensor(points))
        batch_labels.append(label)
    return torch.stack(batch_points), torch.Tensor(batch_labels).long()
