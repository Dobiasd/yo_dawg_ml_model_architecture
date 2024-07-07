from __future__ import annotations

import abc
import collections
import csv
import itertools
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from operator import itemgetter
from pathlib import Path
from typing import Dict, TypeAlias, Any, Callable, TypeVar
from typing import List, Optional, Tuple

import numpy as np
from sklearn import linear_model, svm
from sklearn.neural_network import MLPRegressor

DataPoint: TypeAlias = Dict[str, float]
LabelledDataPoint: TypeAlias = Tuple[float, DataPoint]
DataSet: TypeAlias = List[DataPoint]
LabelledDataSet: TypeAlias = List[LabelledDataPoint]

random.seed(42)

T = TypeVar("T")

decider_type_to_use = "model"  # threshold or model


class ThresholdPolicy(Enum):
    MEAN = auto()
    MEDIAN = auto()
    MIDDLE = auto()


class Decider(ABC):
    @abstractmethod
    def __call__(self: Decider, x: DataPoint) -> bool:
        pass

    @abc.abstractmethod
    def show(self: Decider) -> str:
        pass


class ThresholdDecider(Decider):
    feature: str
    threshold: float

    def __init__(self: ThresholdDecider, feature: str, threshold: float):
        self._feature = feature
        self._threshold = threshold

    def __call__(self: ThresholdDecider, x: DataPoint) -> bool:
        return x[self._feature] < self._threshold

    def show(self: ThresholdDecider) -> str:
        return f"{self._feature} {self._threshold}"


class Predictor(ABC):

    @abstractmethod
    def __init__(self: Predictor, data: LabelledDataSet):
        pass

    @abstractmethod
    def __call__(self: Predictor, x: DataPoint) -> float:
        pass

    @abc.abstractmethod
    def show(self: Predictor) -> str:
        pass

    @staticmethod
    def _datapoint_to_vector(datapoint: DataPoint) -> np.ndarray:
        return np.array([value for key, value in sorted(datapoint.items())])

    @staticmethod
    def _get_data(data: LabelledDataSet) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([Predictor._datapoint_to_vector(point[1]) for point in data]),
                np.array([point[0] for point in data]))


class LinearRegressionPredictor(Predictor):
    def __init__(self: LinearRegressionPredictor, data: LabelledDataSet):
        x, y = Predictor._get_data(data)
        self._model = linear_model.LinearRegression().fit(x, y)

    def __call__(self: LinearRegressionPredictor, x: DataPoint) -> float:
        return self._model.predict([self._datapoint_to_vector(x)])[0]  # type: ignore

    def show(self: LinearRegressionPredictor) -> str:
        return self._model.__class__.__name__


class SVMPredictor(Predictor):
    def __init__(self: SVMPredictor, data: LabelledDataSet):
        x, y = Predictor._get_data(data)
        self._model = svm.SVR().fit(x, y)

    def __call__(self: SVMPredictor, x: DataPoint) -> float:
        return self._model.predict([self._datapoint_to_vector(x)])[0]  # type: ignore

    def show(self: SVMPredictor) -> str:
        return self._model.__class__.__name__


class MLPPredictor(Predictor):
    def __init__(self: MLPPredictor, data: LabelledDataSet):
        x, y = Predictor._get_data(data)
        self._model = MLPRegressor(random_state=42, max_iter=1000).fit(x, y)

    def __call__(self: MLPPredictor, x: DataPoint) -> float:
        return self._model.predict([self._datapoint_to_vector(x)])[0]  # type: ignore

    def show(self: MLPPredictor) -> str:
        return self._model.__class__.__name__


class ModelDecider(Decider):
    def __init__(self: ModelDecider, predictor: Predictor, data: LabelledDataSet, threshold_policy: ThresholdPolicy):
        self._predictor = predictor
        predictions = list(map(self._predictor, list(map(itemgetter(1), data))))
        if threshold_policy == ThresholdPolicy.MEAN:
            self._threshold = float(np.mean(predictions))
        elif threshold_policy == ThresholdPolicy.MEDIAN:
            self._threshold = float(np.median(predictions))
        elif threshold_policy == ThresholdPolicy.MIDDLE:
            self._threshold = float(np.mean(np.min(predictions) + np.max(predictions)))
        else:
            assert False, f"unknown {threshold_policy=}"
        self._repr = f"{self._predictor.__class__.__name__}, {self._threshold}, {threshold_policy=}"

    def __call__(self: ModelDecider, x: DataPoint) -> bool:
        y = self._predictor(x)
        return y < self._threshold

    def show(self: ModelDecider) -> str:
        return self._repr


@dataclass
class TreeNode(ABC):

    @abc.abstractmethod
    def predict(self: TreeNode, x: DataPoint) -> float:
        pass


@dataclass
class Tree(TreeNode):
    decider: Decider
    left: TreeNode
    right: TreeNode

    def predict(self: Tree, x: DataPoint) -> float:
        return self.left.predict(x) if self.decider(x) else self.right.predict(x)


@dataclass
class FixedTreeLeaf(TreeNode):
    output: float

    def predict(self: FixedTreeLeaf, x: DataPoint) -> float:
        return self.output

    def show(self: FixedTreeLeaf):
        return f"{self.output}"


@dataclass
class ModelTreeLeaf(TreeNode):
    predictor: Predictor

    def predict(self: ModelTreeLeaf, x: DataPoint) -> float:
        return self.predictor(x)

    def show(self: ModelTreeLeaf):
        return self.predictor.show()


def load_data(path: Path, target_value_col_name: str) -> LabelledDataSet:
    dataset: LabelledDataSet = []
    with open(path, encoding="UTF-8") as csvfile:
        row: Dict[str, Any]
        for row in csv.DictReader(csvfile):
            row = {k: float(v) for k, v in row.items()}
            target = row[target_value_col_name]
            del row[target_value_col_name]
            dataset.append((target, row))
    return dataset


def split_data(data: LabelledDataSet) -> Tuple[LabelledDataSet, LabelledDataSet]:
    random.shuffle(data)
    split = int(0.8 * len(data))
    return data[:split], data[split:]


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def mean_squared_error(pairs: List[Tuple[float, float]]) -> float:
    return mean([abs(a - b) ** 2 for a, b in pairs])


def root_mean_squared_error(pairs: List[Tuple[float, float]]) -> float:
    return math.sqrt(mean_squared_error(pairs))


def flatten(nested: List[List[T]]) -> List[T]:
    return list(itertools.chain.from_iterable(nested))


def partition(predicate: Callable[[T], bool], values: List[T]) -> Tuple[List[T], List[T]]:
    yes: List[T] = []
    no: List[T] = []
    for value in values:
        (yes if predicate(value) else no).append(value)
    return yes, no


def data_impurity(data: LabelledDataSet) -> float:
    """gini"""
    counts = collections.Counter(point[0] for point in data)
    return 1.0 - sum((counts[target] / float(len(data))) ** 2 for target in counts.keys())


def split_impurity(left: LabelledDataSet, right: LabelledDataSet) -> float:
    p = float(len(left)) / (len(left) + len(right))
    return p * data_impurity(left) + (1 - p) * data_impurity(right)


def best_split(data: LabelledDataSet) -> Optional[Decider]:
    if len(data) < 2:
        return None

    # Sorted to avoid indeterministic behavior in case two deciders bring the same impurity.
    features = sorted(set(flatten(list(map(lambda point: list(point[1].keys()), data)))))

    def evaluate_decider_impurity(d: Decider) -> float:
        return split_impurity(*partition(lambda point: d(point[1]), data))

    deciders: List[Decider]
    if decider_type_to_use == "threshold":
        deciders = [ThresholdDecider(feat, t) for feat in features for t in set(map(lambda p: p[1][feat], data))]
    elif decider_type_to_use == "model":
        predictors = [
            LinearRegressionPredictor(data),
            SVMPredictor(data),
            # MLPPredictor(data),
        ]
        deciders = [ModelDecider(predictor, data, thresholdPolicy) for predictor in predictors for thresholdPolicy in
                    ThresholdPolicy]
    else:
        assert False, f"Unknown {decider_type_to_use=}"
    best_decider = min(deciders, key=evaluate_decider_impurity)
    if evaluate_decider_impurity(best_decider) >= data_impurity(data):
        return None
    return best_decider


def train(data: LabelledDataSet) -> TreeNode:
    assert len(data) > 0
    if len(data) < 2:
        return FixedTreeLeaf(data[0][0])
    decider = best_split(data)
    if not decider:
        if decider_type_to_use == "threshold":
            return FixedTreeLeaf(mean(list(map(itemgetter(0), data))))
        elif decider_type_to_use == "model":
            return ModelTreeLeaf(LinearRegressionPredictor(data))
        assert False, f"Unknown {decider_type_to_use=}"
    data_left, data_right = partition(lambda point: decider(point[1]), data)
    return Tree(decider, train(data_left), train(data_right))


def show_tree(tree: TreeNode) -> str:
    def indent(text: str) -> str:
        return "\n".join(map(lambda line: "    " + line, text.split("\n")))

    if isinstance(tree, FixedTreeLeaf):
        return indent(str(tree.show()))
    elif isinstance(tree, ModelTreeLeaf):
        return indent(str(tree.show()))
    elif isinstance(tree, Tree):
        child = indent(show_tree(tree.left) + "\n" + show_tree(tree.right))
        return f"{tree.decider.show()}\n{child}"
    assert False


def test_rmse(tree: TreeNode, data: LabelledDataSet) -> float:
    targets = list(map(itemgetter(0), data))
    predictions = [tree.predict(datapoint) for target, datapoint in data]
    return root_mean_squared_error(list(zip(targets, predictions)))


def compare_sklearn(data_train: LabelledDataSet, data_test: LabelledDataSet):
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()

    def tabularize(dataset: LabelledDataSet) -> Tuple[List[List[float]], List[float]]:
        features = set(flatten(list(map(lambda point: list(point[1].keys()), data_train))))
        return [[point[1][feature] for feature in features] for point in dataset], list(map(itemgetter(0), dataset))

    x_train, y_train = tabularize(data_train)
    x_test, y_test = tabularize(data_test)
    clf = clf.fit(x_train, y_train)

    # from sklearn.tree import export_text
    # print(export_text(clf))

    print(f'sklearn.tree train RMSE: {root_mean_squared_error(list(zip(y_train, clf.predict(x_train))))}')
    print(f'sklearn.tree Test RMSE: {root_mean_squared_error(list(zip(y_test, clf.predict(x_test))))}')


def main() -> None:
    data = load_data(Path("housing_numerical.csv"), "price")
    data_train, data_test = split_data(data)
    tree = train(data_train)
    print(show_tree(tree))
    print(f'Train RMSE: {test_rmse(tree, data_train)}')
    print(f'Test RMSE: {test_rmse(tree, data_test)}')

    compare_sklearn(data_train, data_test)


if __name__ == '__main__':
    main()
