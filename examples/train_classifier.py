import os
from enum import Enum
import sys

sys.path.append(".")
from lib import (
    HandSignTrainer,
    HandSignClassifier,
    LabeledHandPointsStore,
    NdJsonLabeledHandPointsStore,
    HandSignDataLoggerGUI,
)

from examples.csv_labeled_handpoints_store import CsvLabeledHandPointsStore


SAVE_MODEL_PATH = os.path.join("models", "hand_sign_classifier_weights.pth")


class HandSign(Enum):
    FIST = 0
    LIKE = 1


def get_dataset(store: LabeledHandPointsStore):
    HandSignDataLoggerGUI(store).start()


def train_classifier(store: LabeledHandPointsStore):
    # モデル
    classifier = HandSignClassifier(output_size=len(HandSign))

    # モデル学習オブジェクト
    trainer = HandSignTrainer(store, classifier)
    trainer.train(SAVE_MODEL_PATH)


if __name__ == "__main__":
    # 学習データ書き込み(追記), 読み込みオブジェクト
    store = CsvLabeledHandPointsStore(os.path.join("models", "labeled_hand_sign_dataset.csv"))
    # store = NdJsonLabeledHandPointsStore(os.path.join("models", "labeled_hand_sign_dataset.ndjson"))

    get_dataset(store)  # esc 押下でこの関数は終了する
    train_classifier(store)
