# -*- coding: utf-8 -*-
# @Time     : 2019/7/19 18:42
# @Author   : buracagyang
# @File     : RF.py
# @Software : PyCharm

"""
Describe:
        
"""

import os
import logging
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class RF(object):
    def __init__(self, train_file, test_file, **kwargs):
        self.train_file = train_file
        self.test_file = test_file
        self.features = None
        self.kwargs = kwargs
        logging.info("Selected Model: RF")

    def rf_main(self, each_sample_weight=0.13):
        train_x, train_y = self.process_data(self.train_file)
        test_x, test_y = self.process_data(self.test_file)
        self.features = [x for x in train_x.columns]
        logging.info("******************* DATASET INFO ********************")
        logging.info("TrainSetSize: {}".format(len(train_x)))
        logging.info("TestSetSize: {}".format(len(test_x)))
        logging.info("*****************************************************")

        model = RandomForestClassifier(n_estimators=self.kwargs['n_estimators'],
                                       max_depth=self.kwargs['max_depth'],
                                       bootstrap=self.kwargs['bootstrap'],
                                       min_samples_leaf=self.kwargs['min_samples_leaf'],
                                       min_samples_split=self.kwargs['min_samples_split'],
                                       verbose=self.kwargs['verbose'])
        if each_sample_weight is None:
            sample_weight = None
        else:
            sample_weight = [int(l) + each_sample_weight for l in list(train_y)]
        model.fit(train_x, train_y, sample_weight=sample_weight)
        pred_prob = model.predict_proba(test_x)[:, 1]  # [[0: prob_0, 1: prob_1]]
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
        model_out_path = os.path.join(base_dir, 'model_saved', 'rf')
        other_out_path = os.path.join(base_dir, 'data', 'output')
        os.makedirs(model_out_path, exist_ok=True)
        os.makedirs(other_out_path, exist_ok=True)

        # Dump & Load Model
        if self.kwargs['save_mode']:
            joblib.dump(model, filename=os.path.join(model_out_path, "gbdt_trained.mode"))

        # Plot Features Importance
        if self.kwargs['plot_features_importance']:
            df = pd.DataFrame(list(zip(self.features, model.feature_importances_)),
                              columns=['feature', 'fscore']).sort_values(by=['fscore'])
            df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(8, 10))
            plt.title('RF Feature Importance')
            plt.xlabel('Relative Importance')
            plt.ylabel('Features')
            plt.savefig(os.path.join(other_out_path, 'rf_features_importance.png'))

    @staticmethod
    def process_data(file_path):
        logging.info("loading data from {}.".format(file_path))
        data = pd.read_csv(os.path.join(base_dir, 'data', file_path), encoding='utf-8')
        y = data.label
        x = data.drop('label', axis=1)
        return x, y
