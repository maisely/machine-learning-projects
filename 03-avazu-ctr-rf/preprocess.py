import numpy as np
import pandas as pd
import pickle
import itertools
# from utils import *
import utils

from sklearn.preprocessing import LabelEncoder


train_dtype = {'id': np.uint32,
               'click': np.uint8, 'hour': np.uint32,
               'C1': np.uint32, 'banner_pos': np.uint8,
               'site_id': object, 'site_domain': object,
               'site_category': object, 'app_id': object,
               'app_domain': object, 'app_category': object, 'device_id': object,
               'device_ip': object, 'device_model': object,
               'device_type': np.uint8, 'device_conn_type': np.uint8,
               'C14': np.uint16, 'C15': np.uint16, 'C16': np.uint16,
               'C17': np.uint16, 'C18': np.uint16, 'C19': np.uint16,
               'C20': np.uint16, 'C21': np.uint16}

test_dtype = {'id': np.uint32, 'hour': np.uint32,
              'C1': np.uint32, 'banner_pos': np.uint8,
              'site_id': object, 'site_domain': object,
              'site_category': object, 'app_id': object,
              'app_domain': object, 'app_category': object, 'device_id': object,
              'device_ip': object, 'device_model': object,
              'device_type': np.uint8, 'device_conn_type': np.uint8,
              'C14': np.uint16, 'C15': np.uint16, 'C16': np.uint16,
              'C17': np.uint16, 'C18': np.uint16, 'C19': np.uint16,
              'C20': np.uint16, 'C21': np.uint16}

#--------------------------------------#
# LOAD DATA
#--------------------------------------#
# general path
path = "avazu/"

# load data
data = pd.read_csv(path + "train", usecols=train_dtype.keys(),
                   dtype=train_dtype, low_memory=False)
test_set = pd.read_csv(path + "test", dtype=test_dtype)


# get date features columns
utils.proc_date(data, 'hour')
utils.proc_date(test_set, 'hour')

# convert type for memory
for col in ['hour_prev', 'hour_next']:
    data[col] = data[col].astype('int8')
    test_set[col] = data[col].astype('int8')

# create user_id using device_id, device_ip, device_model
for df in [data, test_set]:
    df['user_id'] = df['device_id'] + df['device_ip'] + df['device_model']

#--------------------------------------#
# SPLIT TRAIN & VALIDATION
#--------------------------------------#
train_set, val_set = utils.split_based_hour(data, col='dt')
print("Train and validation set:")
print(train_set.shape, val_set.shape)

#--------------------------------------#
# TARGET MEAN ENCODING
#--------------------------------------#
for col in ["click_hour", "click_dayofweek", "wk_hour",
            "device_type", "device_conn_type", "banner_pos",
            "site_category", "app_category", "device_type", "C21"]:
    print("processing %s" % col)
    gby = train_set.groupby(col).click.mean()
    train_set['click_gby_' + col] = train_set[col].map(gby)
    val_set['click_gby_' + col] = val_set[col].map(gby)
    test_set['click_gby_' + col] = test_set[col].map(gby)

    # fill na
    click_global_mean = train_set.click.mean()
    val_set['click_gby_' + col].fillna(click_global_mean, inplace=True)
    test_set['click_gby_' + col].fillna(click_global_mean, inplace=True)

# combine anomynized columns
for df in [train_set, val_set, test_set]:
    df['c_combined'] = df['C1'] + df['C14'] + df['C15'] +\
        df['C16'] + df['C17'] + df['C18'] + df['C19'] +\
        df['C20'] + df['C21']

#--------------------------------------#
# COUNT FEATURES
#--------------------------------------#
for col in ['device_ip', 'device_id', 'user_id']:
    count = train_set.groupby(col).index.count()
    train_set['cnt_' + col] = train_set[col].map(count)
    val_set['cnt_' + col] = val_set[col].map(count)
    test_set['cnt_' + col] = test_set[col].map(count)

cols = ['user_id', 'click_hour']
hour_user_cnt = train_set.groupby(cols).agg({'index': 'count'}).reset_index()
hour_user_cnt.columns = [cols[0], cols[1], 'cnt_by_user_hour']

train_set = pd.merge(train_set, hour_user_cnt, how='left', on=cols)
val_set = pd.merge(val_set, hour_user_cnt, how='left', on=cols)
test_set = pd.merge(test_set, hour_user_cnt, how='left', on=cols)

for col in ['cnt_device_ip', 'cnt_device_id', 'cnt_user_id', 'cnt_by_user_hour']:
    val_set[col].fillna(0, inplace=True)
    test_set[col].fillna(0, inplace=True)

#--------------------------------------#
# INTERACTIVE FEATURE PAIR
#--------------------------------------#
for cols in [['app_id', 'site_id'],
             ['app_domain', 'site_domain'],
             ['app_id', 'device_model'],
             ['site_id', 'device_model'],
             ['site_id', 'site_domain']]:
    utils.create_interac(cols[0], cols[1], train_set, val_set, test_set)

#--------------------------------------#
# USER CLICK HISTORY
#--------------------------------------#
cols = ['user_id', 'click_hour']
hour_user_cnt = train_set.groupby(cols).agg({'index': 'count'}).reset_index()
hour_user_cnt.columns = [cols[0], cols[1], 'cnt_by_user_hour']

# user visit counts next hour
hour_user_cnt.columns = ['user_id', 'hour_next', 'cnt_by_user_hour_next']
cols = ['user_id', 'hour_next']

train_set = pd.merge(train_set, hour_user_cnt, how='left', on=cols)
val_set = pd.merge(val_set, hour_user_cnt, how='left', on=cols)
test_set = pd.merge(test_set, hour_user_cnt, how='left', on=cols)

# user visit counts previous hour
hour_user_cnt.columns = ['user_id', 'hour_prev', 'cnt_by_user_hour_prev']
cols = ['user_id', 'hour_prev']

train_set = pd.merge(train_set, hour_user_cnt, how='left', on=cols)
val_set = pd.merge(val_set, hour_user_cnt, how='left', on=cols)
test_set = pd.merge(test_set, hour_user_cnt, how='left', on=cols)

# fill missing values with zero for the count
for col in ['cnt_by_user_hour_next', 'cnt_by_user_hour_prev']:
    train_set[col].fillna(0, inplace=True)
    val_set[col].fillna(0, inplace=True)
    test_set[col].fillna(0, inplace=True)

c_cols = ["C1"] + ["C" + str(i) for i in range(14, 22)]
cat_cols = ['banner_pos', 'device_type', 'device_conn_type'] + c_cols + \
           ['site_id', 'site_domain', 'site_category',
            'app_id', 'app_domain', 'app_category',
            'device_id', 'device_ip', 'device_model', 'user_id']

for df in [train_set, val_set, test_set]:
    utils.convert_cat(df, cat_cols, 'category')

try:
    train_set = train_set.drop(['hour'], axis=1)
    val_set = val_set.drop(['hour'], axis=1)
except:
    print("No index column")

#--------------------------------------#
# LABEL ENCODING
#--------------------------------------#

features = [c for c in train_set.columns if c != 'click']
X_train, y_train = train_set[features], train_set['click'].values
X_val, y_val = val_set[features], val_set['click'].values
X_test = test_set

encode_cols = ['app_id_site_id', 'app_domain_site_domain', 'app_id_device_model',
               'site_id_device_model', 'site_id_site_domain',
               'site_category', 'app_category', 'user_id'] + \
    ['site_id', 'site_domain', 'app_id', 'app_domain',
     'device_id', 'device_ip', 'device_model']

for col in encode_cols:
    print("processing ", col)
    le = LabelEncoder()
    all_label = np.hstack(
        [X_train[col].values, X_val[col].values, X_test[col].values])
    all_label = np.unique(all_label)
    le.fit(list(all_label))
    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])

for df in [X_train, X_val, X_test]:
    utils.convert_cat(df, encode_cols, 'category')
    utils.convert_numtype(df)
