# -*- coding: utf-8 -*-
# @Time     : 2019/7/19 18:29
# @Author   : buracagyang
# @File     : parse_args.py
# @Software : PyCharm

"""
Describe:
        
"""

import argparse


def parse_args():
    """
    包含了XGBOOST, GBDT, RF, GBDT+LR所有模型的参数
    :return:
    """
    parser = argparse.ArgumentParser(description='Manpower Loss Forecast.')
    parser.add_argument('--model', choices=['XGBOOST', 'GBDT', 'RF', 'LR', 'GBDT+LR', 'RF+LR'], default='XGBOOST')
    parser.add_argument('--train_file', type=str, default='train.csv', help='')
    parser.add_argument('--test_file', type=str, default='test.csv', help='')
    parser.add_argument('--ypred_threshold', type=float, default=.5, help='Threshold for predict probability')
    parser.add_argument('--save_mode', type=bool, default=True, help='Whether to save the model')
    parser.add_argument('--plot_features_importance', type=bool, default=True, help='Plot features importance')
    parser.add_argument('--features_map_file', type=str, default='features_map.fmap', help='Features map')

    # GBDT & RF & LR
    parser.add_argument('--n_estimators', type=int, default=25, help='Estimators num in GBDT/RF')
    parser.add_argument('--min_samples_leaf', type=int, default=30, help='Min samples of leaf in GBDT/RF')
    parser.add_argument('--min_samples_split', type=int, default=50, help='')
    parser.add_argument('--verbose', type=int, default=1, help='')

    # XGBOOST
    parser.add_argument('--booster', type=str, default='gbtree', help='Booster type in XGBOOST')
    parser.add_argument('--objective', type=str, default='binary:logistic', help='Objective type in XGBOOST')
    parser.add_argument('--eval_metrix', type=str, default='auc', help='Evaluation metrix during training in XGBOOST')
    parser.add_argument('--max_depth_xgb', type=int, default=8, help='Max depth in XGBOOST')
    parser.add_argument('--lambda_xgb', type=float, default=10.0, help='L2 regularization in XGBOOST')
    parser.add_argument('--subsample_xgb', type=float, default=1.0, help='Subsample ratio in XGBOOST')
    parser.add_argument('--min_child_weight', type=int, default=2, help='')
    parser.add_argument('--eta', type=float, default=0.025, help='Attenuation weight for each step in XGBOOST')
    parser.add_argument('--silent', type=bool, default=True, help='Do not print log message')
    parser.add_argument('--scale_pos_weight', type=int, default=8, help='Useful for unbalanced classes in XGBOOST')
    parser.add_argument('--num_boost_round', type=int, default=50, help='Number of boosting iterations in XGBOOST')

    # GBDT
    parser.add_argument('--learning_rate', type=float, default=.1, help='Learning rate in GBDT')
    parser.add_argument('--max_features', type=str, default='sqrt', help='num of features for the best split in GBDT')
    parser.add_argument('--max_depth_gbdt', type=int, default=10, help='Max depth in GBDT')
    parser.add_argument('--subsample_gbdt', type=float, default=1.0, help='Subsample ratio in GBDT')

    # RF
    parser.add_argument('--max_depth_rf', type=int, default=10, help='Max depth in RF')
    parser.add_argument('--bootstrap', type=bool, default=True, help='Max depth in RF')

    # LR
    parser.add_argument('--penalty', type=str, default='l2', help='')
    parser.add_argument('--dual', type=bool, default=False, help='')
    parser.add_argument('--tol', type=float, default=1e-4, help='')
    parser.add_argument('--C', type=float, default=1.0, help='')
    parser.add_argument('--fit_intercept', type=bool, default=True, help='')
    parser.add_argument('--intercept_scaling', type=int, default=1, help='')
    parser.add_argument('--random_state', type=int, default=None, help='')
    parser.add_argument('--solver', type=str, default='sag', help='')
    parser.add_argument('--max_iter', type=int, default=100, help='')
    parser.add_argument('--multi_class', choices=['warn', 'ovr', 'multinomial', 'auto'], default='warn')
    parser.add_argument('--warm_start', type=bool, default=False, help='')
    parser.add_argument('--n_jobs', type=int, default=None, help='')
    parser.add_argument('--l1_ratio', type=float, default=None, help='')

    return parser.parse_args()
