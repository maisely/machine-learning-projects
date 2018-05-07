import numpy as np
import pandas as pd

# model
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# nlp package spacy
import spacy
nlp = spacy.load('en')


def fill_missing_performer(row, performer_cols, exist_performers):
    """
    Use spaCy to extract names
    Impute if exist in historical performers
    """
    title = row['event_title']

    # replace dash with comma for better results
    title = title.replace("-", ",").replace("&", "and")
    # tokenize the text for entities
    doc = nlp(title)
    entities = [t.text for t in doc.ents if t.label_ == 'PERSON']

    # extract the list of performers in the performer columns
    filled_perf = row[performer_cols].values.tolist()
    # check if the entities are in the historical records
    entities = [t for t in entities
                if t in exist_performers and t not in filled_perf]

    filled_perf = [t for t in filled_perf if pd.notnull(t)]
    result = filled_perf + entities

    if len(result) < 4:
        result += ["None"] * (4 - len(result))

    return tuple(result[:4])


def add_date_ft(raw):
    """
    Add day of week and hour for event and listing date
    :param raw: original dataframe
    """
    # calculate number of days between listing and event date
    raw['days_till_event'] = (raw['event_datetime'] -
                              raw['listing_date']).apply(lambda x: x.days)

    raw['weeks_till_event'] = np.ceil(raw['days_till_event'] / 7).astype(int)

    # day of week of the event
    raw['event_dow'] = raw['event_datetime'].dt.dayofweek
    raw['event_hour'] = raw['event_datetime'].dt.hour

    # day of week of the listing
    raw['listing_dow'] = raw['listing_date'].dt.dayofweek

    return raw


def encode_one_hot(raw, performer_cols):
    """
    Convert performer columns into one-hot encoding
    :param raw: original dataframe
    :param performer_cols: col performer_1, performer_2, etc.
    :return: dataframe with each performer name as a column
    """
    # one-hot encoding all the performers
    raw[performer_cols] = raw[performer_cols].fillna("None")
    raw['performers'] = raw[performer_cols].values.tolist()

    mlb = MultiLabelBinarizer()
    one_hot_df = mlb.fit_transform(raw.pop('performers'))
    one_hot_df = pd.DataFrame(one_hot_df,
                              columns=mlb.classes_).drop(['None'], axis=1)

    raw_encoded = pd.concat([raw, one_hot_df], axis=1)
    raw_encoded = raw_encoded.drop(performer_cols, axis=1)

    # encoded columns for performer in the binary form of 011000 for example
    raw_encoded['performers'] = raw_encoded[one_hot_df.columns[0]].astype(str)
    for col in one_hot_df.columns[1:]:
        raw_encoded['performers'] += raw_encoded[col].astype(str)

    return raw_encoded, one_hot_df.columns


def group_lag_feature(df, target_col, fld, minus, freq="days"):
    """
    Create lag and differences feature at target level
    """
    # column names
    colname = 'last_%d_%s_%s_%s' % (minus, freq, target_col, fld)
    colname_diff = 'last_%d_%s_%s_diff_%s' % (minus, freq, target_col, fld)
    colname_median = 'median_%s_%s_by_%s' % (fld, target_col, freq)

    # tag features by group
    ts_group = df.groupby([target_col,
                           freq + '_till_event']
                          )[fld].median().reset_index().dropna()
    ts_group = ts_group.sort_values(by=[
        target_col, freq + '_till_event'], ascending=[True, False])

    # calculate lag features
    ts_group[colname] = ts_group[fld].shift(minus)
    # ts_group[colname_diff] = ts_group[colname].diff(minus)
    ts_group.rename(columns={fld: colname_median}, inplace=True)
    ts_group = ts_group.dropna().reset_index(drop=True)

    # merge with original
    df = df.copy()
    try:
        df = df.drop([colname_median], axis=1)
    except ValueError:
        pass

    df = df.merge(ts_group, how='left', on=[target_col, freq + '_till_event'])

    # fill in missing values
    for col in [colname, colname_median]:
        df[col] = df.groupby(['event_id'])[col].ffill()
        df[col] = df.groupby([target_col])[col].ffill()
        df[col] = df.groupby([target_col])[col].bfill()
        df[col] = df.groupby(['performers',
                              freq + '_till_event']
                             )[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df.groupby(['taxonomy',
                              freq + '_till_event']
                             )[col].transform(lambda x: x.fillna(x.mean()))

    return df


def fill_missing_lag(df, col, freq="days"):
    """
    Impute missing lag features using back and forward filling
    """
    df = df.copy()
    df[col] = df.groupby(['event_id'])[col].ffill()
    df[col] = df.groupby(['event_id'])[col].bfill()
    df[col] = df.groupby(['performers',
                          freq+'_till_event']
                         )[col].transform(lambda x: x.fillna(x.mean()))

    df[col] = df.groupby(['taxonomy',
                          freq+'_till_event']
                         )[col].transform(lambda x: x.fillna(x.mean()))
    return df


def granular_lag_feature(df, col):
    """
    Create lag feature by event_id
    """
    df = df.sort_values(['event_id', 'event_datetime', 'listing_date'])

    for i in range(1, 4):
        colname = 'last_%d_day_%s' % (i, col)
        colname_diff = 'last_%d_day_diff_%s' % (i, col)

        # last_day: the amount of tickets that a event had in the previous week
        df[colname] = df.groupby(['event_id'])[col].shift(i)
        # fill in missing value
        df = fill_missing_lag(df, colname)

        # last_day_diff: the difference between the amount of tickets
        # in the previous week and the week before it (t-1 - t-2)
        df[colname_diff] = df.groupby(['event_id'])[colname].diff()
        df = fill_missing_lag(df, colname_diff)

    return df


def process_data(raw, performer_cols, exist_performers):
    """
    Clean, add engineered feature and split original data
    :param raw: original dataframe
    :param performer_cols: col performer_1, performer_2, etc.
    :param exist_performers: list of existing performers
    :return: train, validation and testing datasets for model
    """
    print("Filling in missing data...")
    # fill in missing taxonomy
    raw.loc[raw.event_title.str.contains(
        "Los Angeles Philharmonic"), "taxonomy"] = "Classical Orchestral"

    # fill in missing performers
    raw['performer_1'], raw['performer_2'],\
        raw['performer_3'], raw['performer_4'] = \
        zip(*raw[performer_cols + ['event_title']]
            .apply(lambda x:
            fill_missing_performer(x, performer_cols,
                                   exist_performers), axis=1))

    # ------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------

    # add date features
    print("Adding date features...")
    raw = add_date_ft(raw)

    # one-hot encoding all the performers
    print("One-hot encoding performers...")
    raw_encoded, one_hot_cols = encode_one_hot(raw, performer_cols)

    # create lag and differences feature at target level
    print("Adding time lag features...")
    for i in range(1, 4):
        for col in ['taxonomy', 'performers']:
            raw_encoded = group_lag_feature(
                raw_encoded, col, 'tickets_listed', i)
            raw_encoded = group_lag_feature(
                raw_encoded, col, 'mean_listing_price', i)

    # create lag and differences feature for event id
    for col in ['tickets_listed', 'mean_listing_price']:
        raw_encoded = granular_lag_feature(raw_encoded, col)
        med_cols = ['last_1_day_'+col, 'last_2_day_'+col, 'last_3_day_'+col]
        raw_encoded['rolling_'+col] = raw_encoded[med_cols].median(axis=1)

    # ------------------------------------------------
    # Train-Test-Val Split
    # -----------------------------------------------

    print("Splitting data into train, validation, test...")

    # split data into training and testing
    data = raw_encoded[~raw_encoded['mean_listing_price'].isnull()]
    test = raw_encoded[raw_encoded['mean_listing_price'].isnull()]

    # sort data
    data = data.sort_values(['event_id', 'event_datetime', 'listing_date'])
    data = data.dropna()
    test = test.sort_values(['event_id', 'event_datetime', 'listing_date'])

    # create validating dataset
    test_events = list([title for title in test.event_id.unique()])
    val = pd.DataFrame()
    for event in test_events:
        tmp = data[data.event_id == event]
        nrow = tmp.shape[0]
        if nrow <= 5:
            val = pd.concat([val, tmp], axis=0)
        elif nrow <= 30:
            x = int(np.round(nrow * .6))
            val = pd.concat([val, tmp.tail(x)], axis=0)
        else:
            x = int(np.round(nrow * .4))
            val = pd.concat([val, tmp.tail(x)], axis=0)

    # create training dataset that excludes validating rows
    val_idx = val.event_listing_date_id.values
    train = data[~data.event_listing_date_id.isin(val_idx)]

    # check the size
    assert len(train) + len(val) == len(data)

    # sort data
    train = train.sort_values(['event_id', 'event_datetime', 'listing_date'])
    val = val.sort_values(['event_id', 'event_datetime', 'listing_date'])
    test = test.sort_values(['event_id', 'event_datetime', 'listing_date'])

    # check dimension
    print("Dimension of training, validation and testing set")
    print(train.shape, val.shape, test.shape)
    print("Done processing!")

    return data, train, val, test, one_hot_cols


if __name__ == '__main__':

    # load data
    data_path = 'assessment_data.tsv'
    raw_data = pd.read_csv(data_path, sep="\t",
                           parse_dates=['listing_date', 'event_datetime'])

    PERFORMER_COLS = [col for col in raw_data.columns
                      if col.startswith('performer_')]

    EXIST_PERFORMERS = list(set([
        item for lst in raw_data[PERFORMER_COLS].values.tolist()
        for item in lst]))[1:]
