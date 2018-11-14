import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator
from functools import reduce
from collections import Counter


def mean(vector):
    return sum(vector) / len(vector)


def stand_deviation(vector):
    average = mean(vector)
    return math.sqrt((sum((x - average) ** 2 for x in vector)) / max(1, len(vector) - 1))


def createFeatureClassDictionary(X, Y):
    featureClassDict = dict()
    for x, y in zip(X, Y):
        if y in featureClassDict:
            featureClassDict[y] = np.append(featureClassDict[y], [x], axis=0)
        else:
            featureClassDict[y] = np.array([x])
    return featureClassDict


class NaiveBayesNominal:
    def __init__(self):
        self.classes_ = None
        self.model = dict()
        self.y_prior = []
        self.probabilities = {}
        self.class_probabilities = {}
        self.classes = []
        self.featureProbabilities = []
        self.featureClassDict = {}

    def fit(self, X, y):
        self.featureClassDict = createFeatureClassDictionary(X, y)
        for key in self.featureClassDict.keys():
            self.classes.append(key)
            self.class_probabilities[key] = len(self.featureClassDict.get(key)) / len(X)
            self.featureProbabilities = [{key: item / len(line) for key, item in Counter(line).items()} for line in
                                         self.featureClassDict.get(key).T]
            self.probabilities[key] = self.featureProbabilities
        self.classes.sort()

    def predict_proba(self, features):
        result = []
        for feature in features:
            probabilities = []
            for y in self.classes:
                class_probability = self.class_probabilities[y]
                for value, featureValue in enumerate(feature):
                    value_probability = self.probabilities[y][value][featureValue]
                    class_probability = class_probability * value_probability
                probabilities.append(class_probability)
            result.append(probabilities)
        return np.array(result)

    def predict(self, features):
        result = self.predict_proba(features)
        return [list(row).index(max(row)) for row in result]


class NaiveBayesGaussian:
    def __init__(self):
        self.mean_coef = {}
        self.stdev_coef = {}
        self.classes = []
        self.class_probabilities = {}
        self.featureClassDict = {}

    def fit(self, X, y):
        self.featureClassDict = createFeatureClassDictionary(X, y)
        for key in self.featureClassDict.keys():
            self.class_probabilities[key] = len(self.featureClassDict.get(key)) / len(X)
            means = []
            stand_deviations = []
            values = np.array(self.featureClassDict.get(key))
            self.classes.append(key)
            for line in values.T:
                means.append(mean(line))
                stand_deviations.append(stand_deviation(line))
            self.mean_coef[key] = means
            self.stdev_coef[key] = stand_deviations
        self.classes.sort()
        return self

    def predict_proba(self, features):
        result = []
        for x in features:
            probabilities = []
            for y in self.classes:
                values = norm.pdf(x, loc=self.mean_coef[y], scale=self.stdev_coef[y])
                probability = self.class_probabilities[y] * reduce(lambda x, y: x * y, values)
                probabilities.append(probability)
            result.append(probabilities)
        return np.array(result)

    def predict(self, X):
        x = self.predict_proba(X)
        return [list(row).index(max(row)) for row in x]


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        self.is_cat = is_cat
        self.m = m
        self.nb = None

    def fit(self, X, y):
        if self.is_cat:
            self.nb = NaiveBayesNominal()
        else:
            self.nb = NaiveBayesGaussian()
        self.nb.fit(X, y)

    def predict_proba(self, X):
        return self.nb.predict_proba(X)

    def predict(self, X):
        return self.nb.predict(X)
