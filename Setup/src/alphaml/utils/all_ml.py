import pandas as pd
import numpy as np
import logging
from ..utils.utils import scale_pos_weight_
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier,
                              RandomForestClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC, SVC
from ..utils.binary_tabnet import Classifier as TabNet
from ..utils.metrics import BinaryClassificationMetrics, BinaryClassificationLosses
import warnings
from collections import defaultdict

# ***** Calculate AUC and Accuracy *****


def run_ml_models(x_train,
                  x_test,
                  y_train,
                  y_test,
                  path,
                  random_state,
                  sampling_method,
                  col_label
                  ):
    warnings.filterwarnings("ignore")
    
    # Define model parameters

    clf_abc = AdaBoostClassifier(estimator=ExtraTreeClassifier(criterion="log_loss",
                                                               max_depth=4,
                                                               class_weight='balanced'
                                                               ),
                                 learning_rate=0.01,
                                 n_estimators=200,
                                 random_state=random_state
                                 )
    clf_cbc = CatBoostClassifier(learning_rate=0.0075,
                                 depth=6,
                                 subsample=0.75,
                                 colsample_bylevel=0.75,
                                 l2_leaf_reg=2,
                                 random_strength=2.5,
                                 iterations=250,
                                 verbose=False,
                                 random_state=random_state,
                                 auto_class_weights='Balanced'
                                 )
    clf_etc = ExtraTreesClassifier(criterion='log_loss',
                                   max_depth=6,
                                   n_estimators=250,
                                   random_state=random_state,
                                   class_weight='balanced'
                                   )
    clf_gnb = GaussianNB(
                        )
    clf_gbc = GradientBoostingClassifier(learning_rate=0.0075,
                                         max_depth=6,
                                         n_estimators=250,
                                         subsample=0.75,
                                         random_state=random_state
                                         )
    clf_hgb = HistGradientBoostingClassifier(learning_rate=0.0075,
                                             max_depth=6,
                                             max_iter=250,
                                             l2_regularization=2.5,
                                             random_state=random_state,
                                             class_weight='balanced'
                                             )
    clf_knn = KNeighborsClassifier(
                                   )
    clf_lgb = LGBMClassifier(learning_rate=0.0075,
                             max_depth=6,
                             subsample=0.75,
                             colsample_bytree=0.75,
                             reg_alpha=2.5,
                             reg_lambda=2,
                             min_child_weight=2.5,
                             n_estimators=250,
                             verbose=-1,
                             random_state=random_state,
                             class_weight='balanced'
                             )
    clf_mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128),
                            random_state=random_state,
                            max_iter=500,
                            verbose=0
                            )
    clf_svn = NuSVC(random_state=random_state,
                    probability=True,
                    class_weight='balanced'
                    )
    clf_tab = TabNet(mask_type='sparsemax',
                     seed=random_state,
                     verbose=0,
                     n_d=22,
                     n_a=22,
                     n_steps=4,
                     gamma=1.2,
                     n_independent=3,
                     n_shared=3,
                     lambda_sparse=0.00005
                     )
    clf_rfc = RandomForestClassifier(criterion='log_loss',
                                     max_depth=6,
                                     n_estimators=250,
                                     random_state=random_state,
                                     class_weight='balanced'
                                     )
    clf_sl1 = SGDClassifier(penalty='l1',
                            random_state=random_state,
                            loss='log_loss',
                            class_weight='balanced'
                            )
    clf_sl2 = SGDClassifier(penalty='l2',
                            random_state=random_state,
                            loss='log_loss',
                            class_weight='balanced'
                            )
    clf_sen = SGDClassifier(penalty='elasticnet',
                            random_state=random_state,
                            loss='log_loss',
                            class_weight='balanced'
                            )
    clf_svc = SVC(probability=True,
                  random_state=random_state,
                  class_weight='balanced'
                  )
    clf_xgb = XGBClassifier(learning_rate=0.0075,
                            max_depth=6,
                            subsample=0.75,
                            colsample_bytree=0.75,
                            gamma=2,
                            reg_alpha=2.5,
                            reg_lambda=2,
                            min_child_weight=2.5,
                            n_estimators=250,
                            verbosity=0,
                            random_state=random_state,
                            scale_pos_weight=scale_pos_weight_(y_train)
                            )
    
    # Create a list of models
    clf_list = [(clf_abc, "Ada Boost"),
                (clf_cbc, "Cat Boost"),
                (clf_etc, "Extra Trees"),
                (clf_gbc, "Gradient Boosting"),
                (clf_gnb, "Gaussian NB"),
                (clf_hgb, "Hist Gradient Boosting"),
                (clf_knn, "KNeighbors"),
                (clf_sen, "Elastic Net"),
                (clf_sl1, "LASSO"),
                (clf_lgb, "Light GBM"),
                (clf_mlp, "Multi-layer Perceptron"),
                (clf_svn, "Nu SVC"),
                (clf_tab, "TabNet"),
                (clf_rfc, "Random Forest"),
                (clf_sl2, "Ridge"),
                (clf_svc, "SVC rbf"),
                (clf_xgb, "XGBoost")
                ]

    test_dict = defaultdict(dict)
    train_dict = defaultdict(dict)
    losses_dict = defaultdict(dict)
    
    for clf, clf_name in clf_list:
        
        clf.fit(x_train.to_numpy(), y_train.to_numpy())
        test_instance = BinaryClassificationMetrics(clf, x_test.to_numpy(), y_test.to_numpy())
        train_instance = BinaryClassificationMetrics(clf, x_train.to_numpy(), y_train.to_numpy())
        loss_instance = BinaryClassificationLosses(clf, x_test.to_numpy(), y_test.to_numpy())
        test_sc = test_instance.fit()
        train_sc = train_instance.fit()
        losses = loss_instance.fit()
        
        for metric, score in test_sc.items():
            test_dict[clf_name]['Test '+metric] = score
        for metric, score in train_sc.items():
            train_dict[clf_name]['Train '+metric] = score
        for metric, score in losses.items():
            losses_dict[clf_name]['Test loss '+metric] = score
    
    test_df = pd.DataFrame(test_dict).T
    train_df = pd.DataFrame(train_dict).T
    losses_df = pd.DataFrame(losses_dict).T
    # Calculate RMS of differences between train and test scores
    drop_metrics = ["AUC", "Cohen's Kappa", "F1 Score", "Jaccard", "MCC", "Average Precision Score"]
    renamed_test_df = test_df.rename(columns=lambda col: col.replace("Test ", "")).drop(drop_metrics, axis=1)
    renamed_train_df = train_df.rename(columns=lambda col: col.replace("Train ", "")).drop(drop_metrics, axis=1)
    renamed_losses_df = losses_df.rename(columns=lambda col: col.replace("Test loss ", "")).drop(drop_metrics, axis=1)
    diff = np.mean(np.square(renamed_train_df-renamed_test_df), axis=1)**0.5
    # Calculate RMS of test scores losses
    loss_ = np.mean(np.square(renamed_losses_df), axis=1)**0.5
    # Calculate final scores by taking geometric mean, and then making negative log2
    final_scores = pd.DataFrame(-1*np.log2((diff*loss_)**0.5))
    final_scores.columns = ['NegLog2-RMSL']  # Negative log2 Root Mean Squared Loss
        
    all_score_df = pd.concat([train_df, test_df, losses_df, final_scores], axis=1).round(decimals=3)
    # test_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_scores_for_all_ML_test.csv')
    # train_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_scores_for_all_ML_train.csv')
    # losses_df.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_losses_for_all_ML.csv')
    # final_scores.to_csv(f'{path}{sampling_method}_{parameter_search}_Train_Test_final_scores_for_all_ML.csv')
    all_score_df.to_csv(f'{path}{col_label}_{sampling_method}_Train_Test_all_scores_for_all_ML.csv')
    logging.info(f"Scores for all ml have been saved in a csv file in alphaML_results folder")
