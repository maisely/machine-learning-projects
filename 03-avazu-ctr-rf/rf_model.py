import numpy as np
import pandas as pd
import pickle
import itertools
# from utils import *
import utils

from sklearn.ensemble import RandomForestClassifier, forest
from sklearn.metrics import log_loss, f1_score, auc, confusion_matrix
from sklearn.externals import joblib
from sklearn.externals.joblib import parallel_backend


def set_rf_samples(n):
    forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(
                                           0, n_samples, n))


def reset_rf_samples():
    forest._generate_sample_indices = (lambda rs, n_samples:
                                       forest.check_random_state(rs).randint(
                                           0, n_samples, n_samples))


if __name__ == '__main__':

    X_train, y_train = utils.load_feather()
    X_val, y_val = utils.load_feather()
    X_test = utils.load_feather(test_df=True)

    reset_rf_samples()
    set_rf_samples(5000000)

    rf = RandomForestClassifier(
        n_estimators=25,
        min_samples_leaf=10,
        min_samples_split=100,
        max_features='sqrt',
        random_state=42,
        criterion='entropy',
        n_jobs=-1, bootstrap=False)

    with parallel_backend('threading'):
        rf.fit(X_train, y_train)
    joblib.dump(rf, 'rf25_3.pkl')

    y_pred_proba_trn = rf.predict_proba(X_train)
    y_pred_proba_val = rf.predict_proba(X_val)
    print("Training Log Loss: ", log_loss(y_train, y_pred_proba_trn))
    print("Validation Log Loss: ", log_loss(y_val, y_pred_proba_val))

    # Parameter Tuning for Number of Trees
    logloss_train = []
    logloss_val = []

    n_estimators_range = [30, 35, 40]
    for n in n_estimators_range:
        print("Using %s trees:" % str(n))
        params['n_estimators'] = n
        m = RandomForestClassifier(**params,
                                   random_state=42, criterion='entropy',
                                   n_jobs=-1, bootstrap=False)
        with parallel_backend('threading'):
            m.fit(X_train, y_train)

        print("Evaluating...")
        loss_trn = log_loss(y_train, m.predict_proba(X_train))
        loss_val = log_loss(y_val, m.predict_proba(X_val))
        print("===================================")
        print("Training Log Loss: ", loss_trn)
        print("Validation Log Loss: ", loss_val)
        print("===================================")
        print("\n")

        logloss_train.append(loss_trn)
        logloss_val.append(loss_val)

    # Feature Selection
    # pick the best model amongst the above
    rf = RandomForestClassifier(
        n_estimators=35,
        min_samples_leaf=10,
        min_samples_split=100,
        max_features='sqrt',
        random_state=42,
        criterion='entropy',
        n_jobs=-1, bootstrap=False)

    with parallel_backend('threading'):
        rf.fit(X_train, y_train)

    # a more simpler model with only the top 30 features
    fi = pd.DataFrame({
        'feature': X_train.columns,
        'imp_score': rf.feature_importances_
    }).sort_values('imp_score', ascending=False)

    selected = fi.iloc[:30, 0].values
    X_filtered_trn = X_train[selected].copy()
    X_filtered_val = X_val[selected].copy()

    reset_rf_samples()
    set_rf_samples(5000000)
    with parallel_backend('threading'):
        rf.fit(X_filtered_trn, y_train)

    y_pred_proba_trn = rf.predict_proba(X_filtered_trn)
    y_pred_proba_val = rf.predict_proba(X_filtered_val)
    print("Training Log Loss: ", log_loss(y_train, y_pred_proba_trn))
    print("Validation Log Loss: ", log_loss(y_val, y_pred_proba_val))
