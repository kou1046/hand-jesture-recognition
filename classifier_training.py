import os
from lib import HandSignTrainer, HandSignClassifier, NdJsonLabeledHandPointsStore

# 学習データ読み込みオブジェクト
store = NdJsonLabeledHandPointsStore(
    os.path.join(os.path.dirname(__file__), "models", "labeled_hand_sign_dataset.ndjson")
)

# モデル
classifier = HandSignClassifier(output_size=2)

# モデル学習オブジェクト
trainer = HandSignTrainer(store, classifier)
trainer.train(os.path.join(os.path.dirname(__file__), "models", "hand_sign_classifier_weights.pth"))
