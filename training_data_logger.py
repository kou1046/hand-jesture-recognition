import os

from lib import HandSignDataLoggerGUI, NdJsonLabeledHandPointsStore

store = NdJsonLabeledHandPointsStore(os.path.join("models", "labeled_hand_sign_dataset.ndjson"))
HandSignDataLoggerGUI(store).start()
