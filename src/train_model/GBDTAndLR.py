# -*- coding: utf-8 -*-
# @Time     : 2019/7/22 11:44
# @Author   : buracagyang
# @File     : GBDTAndLR.py
# @Software : PyCharm

"""
Describe:
        
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class GBDTAndLR(object):
    def __init__(self, train_file, test_file, **kwargs):
        self.train_file = train_file
        self.test_file = test_file
        self.features = None
        self.kwargs = kwargs
        logging.info("Selected Model: GBDT + LR")

    def gbdt_and_lr_main(self, each_sample_weight=0.13, class_weight=None):
        train_x, train_y = self.process_data(self.train_file)
        test_x, test_y = self.process_data(self.test_file)
        self.features = [x for x in train_x.columns]
        logging.info("******************* DATASET INFO ********************")
        logging.info("TrainSetSize: {}".format(len(train_x)))
        logging.info("TestSetSize: {}".format(len(test_x)))
        logging.info("*****************************************************")

        model_gbdt = GradientBoostingClassifier(n_estimators=self.kwargs['n_estimators'],
                                                learning_rate=self.kwargs['learning_rate'],
                                                min_samples_leaf=self.kwargs['min_samples_leaf'],
                                                min_samples_split=self.kwargs['min_samples_split'],
                                                max_depth=self.kwargs['max_depth'],
                                                max_features=self.kwargs['max_features'],
                                                subsample=self.kwargs['subsample'],
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
        gbdt_enc = OneHotEncoder(categories='auto')
        model_gbdt.fit(train_x, train_y, sample_weight=sample_weight)  # train gbdt model
        gbdt_enc.fit(model_gbdt.apply(train_x)[:, :, 0])  # transformation
        model_lr.fit(gbdt_enc.transform(model_gbdt.apply(train_x)[:, :, 0]), train_y)

        pred_prob = model_lr.predict_proba(gbdt_enc.transform(model_gbdt.apply(test_x)[:, :, 0]))[:, 1]
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
        model_out_path = os.path.join(base_dir, 'model_saved', 'gbdt+lr')
        other_out_path = os.path.join(base_dir, 'data', 'output')
        os.makedirs(model_out_path, exist_ok=True)
        os.makedirs(other_out_path, exist_ok=True)

        # Dump & Load Model
        if self.kwargs['save_mode']:
            joblib.dump(model_lr, filename=os.path.join(model_out_path, "gbdt_lr_trained.mode"))

        # Plot ROC curve
        pred_prob_gbdt = model_gbdt.predict_proba(test_x)[:, 1]
        fpr_gbdt, tpr_gbdt, thresholds_gbdt = metrics.roc_curve(test_y, pred_prob_gbdt)
        fpr_gbdt_lr, tpr_gbdt_lr, thresholds_lr = metrics.roc_curve(test_y, pred_prob)

        roc_curve_df = pd.DataFrame(np.array([fpr_gbdt, tpr_gbdt, thresholds_gbdt]).T, columns=['fpr', 'tpr', 'ts'])
        roc_curve_df.to_csv(os.path.join(other_out_path, 'GBDT_roc_curve_df.csv'), index=False, encoding='utf-8')

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_gbdt, tpr_gbdt, label='GBDT')
        plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBDT + LR')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(other_out_path, 'gbdt_lr_roc_curve.png'))

        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_gbdt, tpr_gbdt, label='GBDT')
        plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBDT + LR')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        plt.savefig(os.path.join(other_out_path, 'gbdt_lr_roc_curve(zoomed in at top left).png'))

    @staticmethod
    def process_data(file_path):
        logging.info("loading data from {}.".format(file_path))
        data = pd.read_csv(os.path.join(base_dir, 'data', file_path), encoding='utf-8')
        y = data.label
        x = data.drop('label', axis=1)
        return x, y
