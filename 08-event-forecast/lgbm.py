import numpy as np
import pandas as pd
import time

# model
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from utils import *
import warnings
warnings.filterwarnings("ignore")


# define independent and dependent variables
def split_vars(df, fld, test=False, exclude=[]):
    """Split data into independent and dependent var"""

    target_vars = ['tickets_listed', 'mean_listing_price']
    rm_vars = ['event_id', 'listing_date',
               'event_listing_date_id', 'event_datetime',
               'performers', 'event_hour']

    features = [col for col in df.columns
                if col not in rm_vars + PERFORMER_COLS +
                target_vars + list(one_hot_cols) + exclude]

    # split into independent and dependent variables
    if not test:
        return df[features], df[fld]
    else:
        return df[features]


def cal_score(y, y_pred):
    """Compute Root Mean Squared Error"""
    print("RMSE: ", np.sqrt(mean_squared_error(y, y_pred)))


def concat_result(df, pred, fld_name):
    """Concat the original data and predicted values"""
    return pd.concat([df.reset_index(drop=True),
                      pd.DataFrame({fld_name: pred})], axis=1)


def label_encode(X, X_train, X_val, X_test):
    """
    Label encoding non-numerical values in the dataset
    :param X_train: training data
    :param X_val: validating data
    :param X_test: testing data
    :return: Dataframe with no non-numerical data
    """
    cat_vars = ['taxonomy', 'event_title', 'venue_name']

    for col in cat_vars:
        print("processing ", col)
        le = LabelEncoder()
        all_label = np.hstack([X_train[col].values,
                               X_val[col].values, X_test[col].values])
        all_label = np.unique(all_label)
        le.fit(list(all_label))

        # transform
        X[col] = le.transform(X[col])
        X_train[col] = le.transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])

    return X, X_train, X_val, X_test


if __name__ == '__main__':

    # load data
    data_path = 'assessment_data.tsv'
    raw_data = pd.read_csv(data_path, sep="\t",
                           parse_dates=['listing_date', 'event_datetime'])
    df = raw_data.copy()

    PERFORMER_COLS = [col for col in raw_data.columns
                      if col.startswith('performer_')]

    EXIST_PERFORMERS = list(set([
        item for lst in raw_data[PERFORMER_COLS].values.tolist()
        for item in lst]))[1:]

    # process data
    data, train, val, test, one_hot_cols = \
        process_data(raw_data, PERFORMER_COLS, EXIST_PERFORMERS)
    TAXONOMIES = list(train.taxonomy.unique())

    # split into independent and dependent variables
    target = 'tickets_listed'
    X_train, y_train_1 = split_vars(train, fld=target)
    X_val, y_val_1 = split_vars(val, fld=target)
    X, y_1 = split_vars(data, fld=target)
    X_test = split_vars(test, test=True, fld=target)

    # label encoding string values
    X, X_train, X_val, X_test = label_encode(X, X_train, X_val, X_test)

    # ------------------------------------------------
    # Ticket Listed Model
    # -----------------------------------------------
    # model parameters
    cat_vars = ['taxonomy', 'event_title', 'venue_name']
    max_rounds_tickets = 2000
    params = {
        'num_leaves': 31,
        'max_depth': 5,
        'objective': 'regression',
        'min_data_in_leaf': 10,
        'learning_rate': 0.007,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'metric': 'l2_root',
        'max_bin': 128,
        'num_threads': 8,
        'early_stopping_round': 50,
        'random_state': 0
    }

    # fit data
    print("Fitting for number of tickets listed...")
    dtrain = lgb.Dataset(
        X_train, label=y_train_1,
        categorical_feature=cat_vars)

    dval = lgb.Dataset(
        X_val, label=y_val_1,
        categorical_feature=cat_vars)

    bst = lgb.train(
        params, dtrain, num_boost_round=max_rounds_tickets,
        valid_sets=[dtrain, dval], verbose_eval=200)

    # predict for training and validation
    best_rounds = bst.best_iteration or max_rounds_tickets
    train_pred_1 = bst.predict(X_train, num_iteration=best_rounds)
    val_pred_1 = bst.predict(X_val, num_iteration=best_rounds)

    for taxonomy in TAXONOMIES:
        tmp_df = pd.concat([val, pd.DataFrame({'y': y_val_1,
                                              'yhat': val_pred_1})], axis=1)
        tmp_df = tmp_df[tmp_df.taxonomy == taxonomy]
        print(taxonomy)
        cal_score(tmp_df['y'], tmp_df['yhat'])

    # fit for whole dataset
    dtrain_whole = lgb.Dataset(
        X, label=y_1,
        categorical_feature=cat_vars)

    bst_whole = lgb.train(
        params, dtrain_whole, num_boost_round=best_rounds,
        valid_sets=[dtrain_whole], verbose_eval=200)

    # predict for testing data
    pred_1 = bst_whole.predict(X, num_iteration=best_rounds)
    test_pred_1 = bst_whole.predict(X_test, num_iteration=best_rounds)

    # check for length of prediction
    assert len(test) == len(X_test) == len(test_pred_1)

    # ------------------------------------------------
    # Listing Price Model
    # -----------------------------------------------

    # add predicted number of tickets listed
    X_train = concat_result(X_train, train_pred_1, 'pred_tickets')
    X_val = concat_result(X_val, val_pred_1, 'pred_tickets')
    X_test = concat_result(X_test, val_pred_1, 'pred_tickets')
    X = concat_result(X, pred_1, 'pred_tickets')

    # extract target variable
    ticket_cols = [col for col in train.columns
                   if col.endswith("tickets_listed")]
    exclude_cols = ticket_cols + list(one_hot_cols)
    target = 'mean_listing_price'
    _, y_train_2 = split_vars(train, fld=target, exclude=exclude_cols)
    _, y_val_2 = split_vars(val, fld=target, exclude=exclude_cols)
    _, y_2 = split_vars(data, fld=target)

    # model parameters
    max_rounds = 5000
    params = {
        'num_leaves': 31,
        'max_depth': 6,
        'objective': 'regression',
        'min_data_in_leaf': 10,
        'learning_rate': 0.001,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'metric': 'l2_root',
        'max_bin': 128,
        'num_threads': 8,
        'early_stopping_round': 50,
        'random_state': 0
    }

    print("Fitting for mean listing price...")
    dtrain = lgb.Dataset(
        X_train, label=y_train_2,
        categorical_feature=cat_vars)

    dval = lgb.Dataset(
        X_val, label=y_val_2,
        categorical_feature=cat_vars)

    bst = lgb.train(
        params, dtrain, num_boost_round=max_rounds,
        valid_sets=[dtrain, dval], verbose_eval=200)

    best_rounds = bst.best_iteration or max_rounds
    train_pred_2 = bst.predict(X_train, num_iteration=best_rounds)
    val_pred_2 = bst.predict(X_val, num_iteration=best_rounds)

    # score for each taxonomy
    for taxonomy in TAXONOMIES:
        tmp_df = pd.concat([val, pd.DataFrame(
                           {'y': y_val_2, 'yhat': val_pred_2})],
                           axis=1)
        tmp_df = tmp_df[tmp_df.taxonomy == taxonomy]
        print(taxonomy)
        cal_score(tmp_df['y'], tmp_df['yhat'])

    dtrain_whole = lgb.Dataset(
        X, label=y_2,
        categorical_feature=cat_vars)

    bst_whole = lgb.train(
        params, dtrain_whole, num_boost_round=best_rounds,
        valid_sets=[dtrain_whole], verbose_eval=200)

    test_pred_2 = bst_whole.predict(X_test, num_iteration=best_rounds)

    # ------------------------------------------------
    # Submission
    # -----------------------------------------------

    print("Preparing for submission...")
    tickets_pred = concat_result(test, test_pred_1,
                                 'pred_tickets').reset_index(drop=True)
    pricing_pred = concat_result(test, test_pred_2,
                                 'pred_price').reset_index(drop=True)

    ticket_map = tickets_pred.set_index(
        'event_listing_date_id').to_dict()['pred_tickets']
    price_map = pricing_pred.set_index(
        'event_listing_date_id').to_dict()['pred_price']

    df.loc[df['tickets_listed'].isnull(), 'tickets_listed'] =\
        df['event_listing_date_id'].map(ticket_map)
    df.loc[df['mean_listing_price'].isnull(), 'mean_listing_price'] =\
        df['event_listing_date_id'].map(price_map)

    # write output to file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file = "predictions_%s.tsv" % timestr
    df.to_csv(output_file, sep='\t', index=False)
    print("Model finished! Please check the output in the directory!")
