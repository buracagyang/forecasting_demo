# -*- coding: utf-8 -*-
# @Time     : 2019/7/22 14:02
# @Author   : buracagyang
# @File     : RFAndLR.py
# @Software : PyCharm

"""
Describe:
        
"""

import os
import logging
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class RFAndLR(object):
    def __init__(self, train_file, test_file, **kwargs):
        self.train_file = train_file
        self.test_file = test_file
        self.features = None
        self.kwargs = kwargs
        logging.info("Selected Model: RF + LR")

    def rf_and_lr_main(self, each_sample_weight=0.13, class_weight=None):
        train_x, train_y = self.process_data(self.train_file)
        test_x, test_y = self.process_data(self.test_file)
        self.features = [x for x in train_x.columns]
        logging.info("******************* DATASET INFO ********************")
        logging.info("TrainSetSize: {}".format(len(train_x)))
        logging.info("TestSetSize: {}".format(len(test_x)))
        logging.info("*****************************************************")

        model_rf = RandomForestClassifier(n_estimators=self.kwargs['n_estimators'],
                                          max_depth=self.kwargs['max_depth'],
                                          bootstrap=self.kwargs['bootstrap'],
                                          min_samples_leaf=self.kwargs['min_samples_leaf'],
                                          min_samples_split=self.kwargs['min_samples_split'],
                                          verbose=self.kwargs['verbose'])

        model_lr = LogisticRegression(penalty=self.kwargs['penalty'],
                                      dual=self.kwargs['dual'],
                                      tol=self.kwargs['tol'],
                                      C=self.kwargs['C'],
                                      fit_intercept=self.kwargs['fit_intercept'],
                                      intercept_scaling=self.kwargs['intercept_scaling'],
                                      class_weight=class_weight,
                                      random_state=self.kwargs['random_state'],
                                      solver=self.kwargs['solver'],
                                      max_iter=self.kwargs['max_iter'],
                                      multi_class=self.kwargs['multi_class'],
                                      verbose=self.kwargs['verbose'],
                                      warm_start=self.kwargs['warm_start'],
                                      n_jobs=self.kwargs['n_jobs'],
                                      l1_ratio=self.kwargs['l1_ratio'])

        if each_sample_weight is None:
            sample_weight = None
        else:
            sample_weight = [int(l) + each_sample_weight for l in list(train_y)]

        # Supervised transformation
        rf_enc = OneHotEncoder(categories='auto')
        model_rf.fit(train_x, train_y, sample_weight=sample_weight)  # train rf model
        rf_enc.fit(model_rf.apply(train_x))  # transformation
        model_lr.fit(rf_enc.transform(model_rf.apply(train_x)), train_y)

        pred_prob = model_lr.predict_proba(rf_enc.transform(model_rf.apply(test_x)))[:, 1]
        y_pred = (pred_prob >= self.kwargs['ypred_threshold']) * 1

        logging.info("******************* EVALUATION METRIC ********************")
        logging.info("AUC: %.4f" % metrics.roc_auc_score(test_y, pred_prob))
        logging.info("ACC: %.4f" % metrics.accuracy_score(test_y, y_pred))
        logging.info("Recall: %.4f" % metrics.recall_score(test_y, y_pred))
        logging.info("Precision: %.4f" % metrics.precision_score(test_y, y_pred))
        logging.info("F1-score: %.4f" % metrics.f1_score(test_y, y_pred))
        print("confusion_matrix: ", metrics.confusion_matrix(test_y, y_pred))
        print(metrics.classification_report(test_y, y_pred))
        logging.info("**********************************************************")

        # create path
        model_out_path = os.path.join(base_dir, 'model_saved', 'rf+lr')
        other_out_path = os.path.join(base_dir, 'data', 'output')
        os.makedirs(model_out_path, exist_ok=True)
        os.makedirs(other_out_path, exist_ok=True)

        # Dump & Load Model
        if self.kwargs['save_mode']:
            joblib.dump(model_lr, filename=os.path.join(model_out_path, "rf_lr_trained.mode"))

        # Plot ROC curve
        pred_prob_rf = model_rf.predict_proba(test_x)[:, 1]
        fpr_rf, tpr_rf, _ = metrics.roc_curve(test_y, pred_prob_rf)
        fpr_rf_lr, tpr_rf_lr, _ = metrics.roc_curve(test_y, pred_prob)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf, label='RF')
        plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(other_out_path, 'rf_lr_roc_curve.png'))

        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_rf, tpr_rf, label='RF')
        plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        plt.savefig(os.path.join(other_out_path, 'rf_lr_roc_curve(zoomed in at top left).png'))

    @staticmethod
    def process_data(file_path):
        logging.info("loading data from {}.".format(file_path))
        data = pd.read_csv(os.path.join(base_dir, 'data', file_path), encoding='utf-8')
        y = data.label
        x = data.drop('label', axis=1)
        return x, y