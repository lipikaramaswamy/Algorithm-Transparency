#qii_analysis.py
#Karina Huang, Lipika Ramaswamy

#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#import qii utils functions
from qii_utils import intervene, individualOutcomes, groupDisparity, random_intervene_point, shapley_influence


class QII:
    '''
    class object for qii computation.

    Parameters:
    ===========
    1) train: pd.DataFrame object, training dataset
    2) test: pd.DataFrame object, test dataset
    3) pred: list or array-like, list of predictors
    4) resp: list or array-like, response variable

    Methods:
    ===========
    1) getModel: update classifier, user defined by string
    2) getPrediction: update training and test predictions,
        automatically called in getAccuracy and getF1
    3) getAccuracy: output training and test classification accuracy
    4) getF1: output training and test f1 scores
    5) getUnaryInd: return histories of unary qii on individuals
        by decile score category
    6) getUnaryGrp: return hitories of unary qii on group disparity
        by decile score category
    7) getSetQII: return shapley influence score given subject id and
        set of features
    '''
    def __init__(self, train, test, pred, resp):

        self.train = train
        self.test = test
        self.predictors = pred
        self.response = resp
        self.trainX = self.train[self.predictors]
        self.trainY = self.train[self.response]
        self.testX = self.test[self.predictors]
        self.testY = self.test[self.response]
        self.model = None
        self.trainPred = None
        self.testPred = None

    def getModel(self, type):
        '''fit classifier'''
        if type == 'Logistic Regression':
            self.model = LogisticRegressionCV(cv = 5, solver = 'newton-cg',random_state = 221)
            self.model.fit(self.trainX, self.trainY)
        elif type == 'Decision Tree':
            self.model = DecisionTreeClassifier(random_state = 221)
            self.model.fit(self.trainX, self.trainY)
        elif type == 'SVM':
            self.model = SVC(probability = True, random_state = 221)
            self.model.fit(self.trainX, self.trainY)
        else:
            print('Available Model Class: 1) Logistic Regression, 2) Decision Tree, 3) SVM')

    def getPrediction(self):
        '''update training and test predictions'''
        self.trainPred = self.model.predict(self.trainX)
        self.testPred = self.model.predict(self.testX)

    def getAccuracy(self, training = False):
        '''output training and test accuracies'''
        self.getPrediction()

        if training:
            acc = accuracy_score(self.trainPred, self.trainY)
            print('Training Prediction Accuracy: ', acc)
        else:
            acc = accuracy_score(self.testPred, self.testY)
            print('Test Prediction Accuracy: ', acc)

    def getF1(self, training = False):
        '''output training and test f1 scores'''
        self.getPrediction()

        if training:
            f1 = f1_score(self.trainPred, self.trainY, average = 'weighted')
            print('Training F1 Score: ', f1)
        else:
            f1 = f1_score(self.testPred, self.testY, average = 'weighted')
            print('Test F1 Score: ', f1)

    def getUnaryInd(self):
        '''return average unary qii on individual outcomes by decile score category'''
        low, med, high = individualOutcomes(self.test, self.model, self.predictors)

        return low, med, high

    def getUnaryGrp(self, grp, reference_grp):
        '''return unary qii on group disparity and labels by decile score category'''
        low, med, high, labels = groupDisparity(self.model, self.test, grp, reference_grp, self.predictors)

        return low, med, high, labels

    def getSetQII(self, idx, varList):
        '''return shapley influence measure given subject id and set of features for consideration'''
        shapley = shapley_influence(self.testX, self.model, idx, self.testX, varList, self.predictors)

        return shapley
