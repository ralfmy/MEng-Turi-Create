from helpers import *
import json

class Model:
    def __init__(self, classifier, metrics):
        self.classifier = classifier
        self.metrics = metrics

class Metrics:
    def __init__(self, train_locs=[], test_locs=[], train_size=0, test_size=0, metrics={}):
        self.train_locs = train_locs
        self.test_locs = test_locs
        self.train_size = train_size
        self.test_size = test_size
        self.metrics = metrics
        self.accuracy = metrics["accuracy"]
        self.precision = metrics["precision"]
        self.recall = metrics["recall"]
        self.f1 = metrics["f1_score"]
        self.logloss = metrics["log_loss"]
        self.confusion_matrix = metrics["confusion_matrix"]
        self.roc_curve = metrics["roc_curve"]

    def display(self):
        print(f'training locations: {self.train_locs}   sample size: {self.train_size}')
        print(f'testing locations: {self.test_locs}   sample size: {self.test_size}')
        print(f'accuracy: {round(self.accuracy, 2)}, precision: {round(self.precision, 2)}, recall: '
              f'{round(self.recall, 2)}, f1: {round(self.f1, 2)}, log loss: {round(self.logloss, 2)}')
        print(self.confusion_matrix)
        print()

    def explore(self):
        self.metrics.explore()