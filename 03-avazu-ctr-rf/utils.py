import numpy as np
import pandas as pd
import pickle
import itertools

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, forest
from sklearn.metrics import log_loss, f1_score, auc, confusion_matrix


path = "avazu/"


def load_data():
    """Load data from path"""
    data = pd.read_csv(path + "train")
    test = pd.read_csv(path + "test")
    return data, test


def load_feather(test_df=False):
    """Load dataframes as feather"""
    train = pd.read_feather(path + "train_set")
    val = pd.read_feather(path + "val_set")
    if test_df:
        test = pd.read_feather(path + "test_set")
        return train, val, test
    else:
        return train, val


def save_feather(test_df=False):
    """Save dataframes as feather"""
    train_set.to_feather(path + "train_set")
    val_set.to_feather(path + "val_set")
    if test_df:
        test_set.to_feather(path + "test_set")


def df_dtype(df):
    """Show the datatype of df columns"""
    dict_type = {}
    for col, typ in df.dtypes.items():
        if str(typ) not in dict_type:
            dict_type[str(typ)] = [col]
        else:
            dict_type[str(typ)].append(col)
    return dict_type


def convert_numtype(df):
    """Convert float64 to float32 to save memory"""
    dtype = df_dtype(df)
    try:
        for col in dtype['float64']:
            df[col] = df[col].astype(np.float32)
    except:
        print("No float64 columns")
    try:
        for col in dtype['int64']:
            if col != 'index':
                df[col] = df[col].astype(np.int32)
    except:
        print("No int64 columns")
    # return df_dtype(df)


def convert_cat(df, cols, typ):
    for col in cols:
        df[col] = df[col].astype(typ)


def proc_date(df, date_fld):
    df['date'] = pd.to_datetime(df[date_fld], format="%y%m%d%H")
    for n in ('hour', 'day', 'dayofweek'):
        df["click_" + n] = getattr(df['date'].dt, n)
    df.drop(['date'], axis=1, inplace=True)
    df['hour_prev'] = df['click_hour'] - 1
    df['hour_next'] = df['click_hour'] + 1
    df['wk_hour'] = df['click_dayofweek'] * 24 + df['click_hour']
    df['wk_hour_prev'] = df['wk_hour'] - 1
    df['wk_hour_next'] = df['wk_hour'] + 1

    for col in ['click_hour', 'click_day', 'click_dayofweek',
                'wk_hour', 'wk_hour_prev', 'wk_hour_next']:
        df[col] = df[col].astype('int8')


def split_based_hour(data, col):
    """ Split data based on column hour. Modified to take other columns
    """
    # N = int(0.8*len(data))
    # data = data.sort_values(by=col)-data was ordered chronologically
    train = data[data['click_day'] < 29]
    val = data[data['click_day'] >= 29]
    return train.reset_index(), val.reset_index()


def create_interac(col1, col2, train, val, test):
    # add concatenated columns
    new_col = "_".join([col1, col2])
    train[new_col] = (train[col1] + train[col2]).astype('category')
    val[new_col] = (val[col1] + val[col2]).astype('category')
    test[new_col] = (test[col1] + test[col2]).astype('category')
