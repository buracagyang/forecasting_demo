# -*- coding: utf-8 -*-
# @Time     : 2019/7/19 13:55
# @Author   : buracagyang
# @File     : test_forecast_effect.py
# @Software : PyCharm

"""
Describe:
        
"""

import os
from src.util.log_config import setup_logging
from src.util.parse_args import parse_args
from src.train_model.XGBOOST import XGBOOST
from src.train_model.GBDT import GBDT
from src.train_model.RF import RF
from src.train_model.LR import LR
from src.train_model.GBDTAndLR import GBDTAndLR
from src.train_model.RFAndLR import RFAndLR
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
base_dir = os.path.dirname(__file__)


if __name__ == "__main__":
    setup_logging(os.path.join(base_dir, 'logger.yaml'))

    args = parse_args()

    if args.model == "XGBOOST":
        kws = {
            'booster': args.booster,
            'objective': args.objective,
            'eval_metric': args.eval_metrix,
            'max_depth': args.max_depth_xgb,
            'lambda': args.lambda_xgb,
            'subsample': args.subsample_xgb,
            'min_child_weight': args.min_child_weight,
            'eta': args.eta,
            'silent': args.silent,
            'scale_pos_weight': args.scale_pos_weight,
            'num_boost_round': args.num_boost_round,
            'ypred_threshold': args.ypred_threshold,
            'save_mode': args.save_mode,
            'features_map_file': args.features_map_file,
            'plot_features_importance': args.plot_features_importance
        }
        xgboost = XGBOOST(args.train_file, args.test_file, **kws)
        xgboost.xgboost_main()

    if args.model == "GBDT":
        kws = {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'min_samples_leaf': args.min_samples_leaf,
            'min_samples_split': args.min_samples_split,
            'max_depth': args.max_depth_gbdt,
            'max_features': args.max_features,
            'subsample': args.subsample_gbdt,
            'verbose': args.verbose,
            'ypred_threshold': args.ypred_threshold,
            'save_mode': args.save_mode,
            'plot_features_importance': args.plot_features_importance
        }
        gbdt = GBDT(args.train_file, args.test_file, **kws)
        gbdt.gbdt_main()

    if args.model == "RF":
        kws = {
            'n_estimators': args.n_estimators,
            'min_samples_leaf': args.min_samples_leaf,
            'min_samples_split': args.min_samples_split,
            'max_depth': args.max_depth_rf,
            'bootstrap': args.bootstrap,
            'verbose': args.verbose,
            'ypred_threshold': args.ypred_threshold,
            'save_mode': args.save_mode,
            'plot_features_importance': args.plot_features_importance
        }
        rf = RF(args.train_file, args.test_file, **kws)
        rf.rf_main()

    if args.model == "LR":
        kws = {
            'penalty': args.penalty,
            'dual': args.dual,
            'tol': args.tol,
            'C': args.C,
            'fit_intercept': args.fit_intercept,
            'intercept_scaling': args.intercept_scaling,
            'random_state': args.random_state,
            'solver': args.solver,
            'max_iter': args.max_iter,
            'multi_class': args.multi_class,
            'verbose': args.verbose,
            'warm_start': args.warm_start,
            'n_jobs': args.n_jobs,
            'l1_ratio': args.l1_ratio,
            'ypred_threshold': args.ypred_threshold,
            'save_mode': args.save_mode
            # 'plot_features_importance': args.plot_features_importance
        }
        lr = LR(args.train_file, args.test_file, **kws)
        lr.lr_main(each_sample_weight=None, class_weight={0: 0.1, 1: 0.9})

    if args.model == 'GBDT+LR':
        kws = {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'min_samples_leaf': args.min_samples_leaf,
            'min_samples_split': args.min_samples_split,
            'max_depth': args.max_depth_gbdt,
            'max_features': args.max_features,
            'subsample': args.subsample_gbdt,
            'penalty': args.penalty,
            'dual': args.dual,
            'tol': args.tol,
            'C': args.C,
            'fit_intercept': args.fit_intercept,
            'intercept_scaling': args.intercept_scaling,
            'random_state': args.random_state,
            'solver': args.solver,
            'max_iter': args.max_iter,
            'multi_class': args.multi_class,
            'verbose': args.verbose,
            'warm_start': args.warm_start,
            'n_jobs': args.n_jobs,
            'l1_ratio': args.l1_ratio,
            'ypred_threshold': args.ypred_threshold,
            'save_mode': args.save_mode
        }
        gbdt_lr = GBDTAndLR(args.train_file, args.test_file, **kws)
        gbdt_lr.gbdt_and_lr_main(class_weight={0: 0.1, 1: 0.9})

    if args.model == 'RF+LR':
        kws = {
            'n_estimators': args.n_estimators,
            'min_samples_leaf': args.min_samples_leaf,
            'min_samples_split': args.min_samples_split,
            'max_depth': args.max_depth_rf,
            'bootstrap': args.bootstrap,
            'penalty': args.penalty,
            'dual': args.dual,
            'tol': args.tol,
            'C': args.C,
            'fit_intercept': args.fit_intercept,
            'intercept_scaling': args.intercept_scaling,
            'random_state': args.random_state,
            'solver': args.solver,
            'max_iter': args.max_iter,
            'multi_class': args.multi_class,
            'verbose': args.verbose,
            'warm_start': args.warm_start,
            'n_jobs': args.n_jobs,
            'l1_ratio': args.l1_ratio,
            'ypred_threshold': args.ypred_threshold,
            'save_mode': args.save_mode
        }
        rf_lr = RFAndLR(args.train_file, args.test_file, **kws)
        rf_lr.rf_and_lr_main(class_weight={0: 0.1, 1: 0.9})
