from __future__ import annotations
import sys
import csv


sys.path.append(".")
from lib import LabeledHandPointsStore, LabeledHandPoints, HandPoints, Point


class CsvLabeledHandPointsStore(LabeledHandPointsStore):
    """
    csvで学習データを書き込み, 読み込み担当するクラス.
    別の形式で永続化したい場合はこのクラスのように
    LabeledHandPointsStoreを継承したクラスを定義しadd, loadメソッドをオーバーライドする.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def add(self, labeled_handpoints: LabeledHandPoints) -> None:
        label = labeled_handpoints.label

        xs = [point.x for point in labeled_handpoints.handpoints.values]
        ys = [point.y for point in labeled_handpoints.handpoints.values]
        values = xs + ys

        # x, y, labelをばらし，1次元化して保存
        row = [label] + values

        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def load(self) -> list[LabeledHandPoints]:
        with open(self.file_path) as f:
            rows = f.readlines()
        dataset: list[LabeledHandPoints] = []
        for row in rows:
            row = row.split(",")
            label = int(row[0])
            xs = [float(cell.replace("\n", "")) for cell in row[1:22]]
            ys = [float(cell.replace("\n", "")) for cell in row[22:]]
            hand_points = HandPoints([Point(x, y) for x, y in zip(xs, ys)])
            labeled_handpoints = LabeledHandPoints(label, hand_points)
            dataset.append(labeled_handpoints)
        return dataset
