import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from ml_tools.utils.logger import get_logger


class SimpleWatcher(object):
    def __init__(self, name, default_value=0, order="ascending", patience=10):
        self.name = name
        self.default_value = default_value
        self.order = order
        self.data = []
        self.jst = timezone(timedelta(hours=9))
        self.patience = patience
        self.counter = 0
        self.best_score = default_value
        self.__count = 0
        self.__is_best = False
        self.__min = sys.maxsize
        self.__max = -1 * sys.maxsize
        self.__mean = 0

    @property
    def early_stop(self):
        return self.counter > self.patience

    @property
    def is_best(self):
        return self.__is_best

    def put(self, x):
        timestamp = datetime.now(self.jst)
        self.data.append((x, timestamp))
        self.data = self.data[-1000:]
        self.__count += 1

        if self.order == "ascending":
            if self.best_score < x:
                self.best_score = x
                self.counter = 0
                self.__is_best = True
            else:
                self.counter += 1
                self.__is_best = False
        if self.order == "descending":
            if x < self.best_score:
                self.best_score = x
                self.counter = 0
                self.__is_best = True
            else:
                self.counter += 1
                self.__is_best = False

        if x < self.__min:
            self.__min = x
        if self.__max < x:
            self.__max = x
        self.__mean = (1 / self.__count + 1e-10) * ((self.__count - 1) * self.__mean + x)

    @property
    def mean(self):
        return self.__mean

    @property
    def max(self):
        return self.__max

    @property
    def min(self):
        return self.__min

    def fps(self):
        if len(self.data) < 1:
            return self.default_value
        m = np.mean(np.diff([x[1] for x in self.data]))
        return 1.0 / (m.seconds + 1e-10)


class LossWatcher(SimpleWatcher):
    def __init__(self, name, patience=10):
        super().__init__(name, default_value=sys.maxsize, order="descending", patience=patience)


class AucWatcher(object):
    def __init__(self, name, threshold=0.5, patience=10):
        self.name = name
        self.threshold = threshold
        self.data = []
        self.jst = timezone(timedelta(hours=9))
        self.patience = patience
        self.counter = 0
        self.best_auc = -1
        self.__is_best = False
        self.__min_eval_count = 4
        self.logger = get_logger("AucWatcher")

    @property
    def early_stop(self):
        return self.counter > self.patience

    @property
    def is_best(self):
        return self.__is_best

    @property
    def precision(self):
        if len(self.data) < self.__min_eval_count:
            return -1
        y_pred = [int(self.threshold < item[0]) for item in self.data]
        y_true = [item[1] for item in self.data]
        score = precision_score(y_true, y_pred)
        return score

    @property
    def recall(self):
        if len(self.data) < self.__min_eval_count:
            return -1
        y_pred = [int(self.threshold < item[0]) for item in self.data]
        y_true = [item[1] for item in self.data]
        score = recall_score(y_true, y_pred)
        return score

    @property
    def auc(self):
        if len(self.data) < self.__min_eval_count:
            return -1
        y_pred = [item[0] for item in self.data]
        y_true = [item[1] for item in self.data]
        try:
            score = roc_auc_score(y_true, y_pred)
            return score
        except Exception as ex:
            self.logger.exception(ex)
            return -1

    def put(self, x, y):
        timestamp = datetime.now(self.jst)

        if isinstance(x, list) or isinstance(x, np.ndarray):
            if x.ndim == 0:
                x = [x]
            if y.ndim == 0:
                y = [y]
            for _x, _y in zip(x, y):
                self.data.append((_x, _y, timestamp))
        else:
            self.data.append((x, y, timestamp))

        if len(self.data) < self.__min_eval_count:
            self.__is_best = False
            return

        auc = self.auc
        if self.best_auc < auc:
            self.best_auc = auc
            self.counter = 0
            self.__is_best = True
        else:
            self.counter += 1
            self.__is_best = False
