# -*- coding: utf-8 -*-
# @Time     : 2019/7/19 13:49
# @Author   : buracagyang
# @File     : XGBOOST.py
# @Software : PyCharm

"""
Describe:
        
"""

import os
import logging
import operator
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class XGBOOST(object):
    def __init__(self, train_file, test_file, **kwargs):
        self.train_file = train_file
        self.test_file = test_file
        self.features = None
        self.num_boost_round = kwargs['num_boost_round']
        self.ypred_threshold = kwargs['ypred_threshold']
        self.save_mode = kwargs['save_mode']
        self.features_map_file = kwargs['features_map_file']
        self.plot_features_importance = kwargs['plot_features_importance']
        kwargs.pop('num_boost_round')
        kwargs.pop('ypred_threshold')
        kwargs.pop('save_mode')
        kwargs.pop('features_map_file')
        kwargs.pop('plot_features_importance')
        self.params = kwargs
        logging.info("Selected Model: XGBOOST")

    def xgboost_main(self):
        train_x, train_y = self.process_data(self.train_file)
        test_x, test_y = self.process_data(self.test_file)
        self.features = [x for x in train_x.columns]
        logging.info("******************* DATASET INFO ********************")
        logging.info("TrainSetSize: {}".format(len(train_x)))
        logging.info("TestSetSize: {}".format(len(test_x)))
        logging.info("*****************************************************")

        data_train = xgb.DMatrix(train_x, label=train_y)
        data_test = xgb.DMatrix(test_x)

        evals = [(data_train, 'train')]
        booster = xgb.train(params=self.params, dtrain=data_train, num_boost_round=self.num_boost_round, evals=evals)
        pred_prob = booster.predict(data_test)
        # 设置阈值，输出一些评价指标
        y_pred = (pred_prob >= self.ypred_threshold) * 1

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
        model_out_path = os.path.join(base_dir, 'model_saved', 'xgboost')
        other_out_path = os.path.join(base_dir, 'data', 'output')
        os.makedirs(model_out_path, exist_ok=True)
        os.makedirs(other_out_path, exist_ok=True)

        # create features map
        self.create_features_map()

        # Dump & Load Model
        if self.save_mode:
            booster.save_model(os.path.join(model_out_path, "xgb_trained.model"))
            booster.dump_model(os.path.join(model_out_path, "xgb.model.txt"),
                               fmap=os.path.join(base_dir, 'data', 'output', self.features_map_file))

        # Plot Features Importance
        if self.plot_features_importance:
            importance = booster.get_fscore(fmap=os.path.join(base_dir, 'data', 'output', self.features_map_file))
            importance = sorted(importance.items(), key=operator.itemgetter(1))

            df = pd.DataFrame(importance, columns=['feature', 'fscore'])
            df['fscore'] = df['fscore'] / df['fscore'].sum()
            df.to_csv(os.path.join(other_out_path, 'features_importance.csv'), index=False, encoding='utf-8')

            df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(8, 10))
            plt.title('XGBoost Feature Importance')
            plt.xlabel('Relative Importance')
            plt.ylabel('Features')
            plt.savefig(os.path.join(other_out_path, 'xgb_features_importance.png'))

    @staticmethod
    def process_data(file_path):
        logging.info("loading data from {}.".format(file_path))
        data = pd.read_csv(os.path.join(base_dir, 'data', file_path), encoding='utf-8')
        y = data.label
        x = data.drop('label', axis=1)
        return x, y

    def create_features_map(self):
        with open(os.path.join(base_dir, 'data', 'output', self.features_map_file), 'w') as f:
            for i, feat in enumerate(self.features):
                f.write('{0}\t{1}\tq\n'.format(i, feat))
