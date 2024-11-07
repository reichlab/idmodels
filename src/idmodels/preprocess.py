import fnmatch

import pandas as pd

from timeseriesutils import featurize
from data_pipeline.utils import get_holidays


def create_features_and_targets(df, incl_level_feats, max_horizon, curr_feat_names = []):
    '''
    Create features and targets for prediction
    
    Parameters
    ----------
    df: pandas dataframe
      data frame with data to "featurize"
    incl_level_feats: boolean
      include features that are a measure of local level of the signal?
    max_horizon: int
      maximum forecast horizon
    curr_feat_names: list of strings
      list of names of columns in `df` containing existing features
    
    Returns
    -------
    tuple with:
    - the input data frame, augmented with additional columns with feature and
      target values
    - a list of all feature names, columns in the data frame
    '''
    
    # current features; will be updated
    feat_names = curr_feat_names
    
    # one-hot encodings of data source, agg_level, and location
    for c in ['source', 'agg_level', 'location']:
        ohe = pd.get_dummies(df[c], prefix=c)
        df = pd.concat([df, ohe], axis=1)
        feat_names = feat_names + list(ohe.columns)
    
    # season week relative to christmas
    df = df.merge(
            get_holidays() \
                .query("holiday == 'Christmas Day'") \
                .drop(columns=['holiday', 'date']) \
                .rename(columns={'season_week': 'xmas_week'}),
            how='left',
            on='season') \
        .assign(delta_xmas = lambda x: x['season_week'] - x['xmas_week'])
    
    feat_names = feat_names + ['delta_xmas']
    
    # features summarizing data within each combination of source and location
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'windowed_taylor_coefs',
                'args': {
                    'columns': 'inc_trans_cs',
                    'taylor_degree': 2,
                    'window_align': 'trailing',
                    'window_size': [4, 6],
                    'fill_edges': False
                }
            },
            {
                'fun': 'windowed_taylor_coefs',
                'args': {
                    'columns': 'inc_trans_cs',
                    'taylor_degree': 1,
                    'window_align': 'trailing',
                    'window_size': [3, 5],
                    'fill_edges': False
                }
            },
            {
                'fun': 'rollmean',
                'args': {
                    'columns': 'inc_trans_cs',
                    'group_columns': ['location'],
                    'window_size': [2, 4]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'lag',
                'args': {
                    'columns': ['inc_trans_cs'] + new_feat_names,
                    'lags': [1, 2]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # add forecast targets
    df, new_feat_names = featurize.featurize_data(
        df, group_columns=['source', 'location'],
        features = [
            {
                'fun': 'horizon_targets',
                'args': {
                    'columns': 'inc_trans_cs',
                    'horizons': [(i + 1) for i in range(max_horizon)]
                }
            }
        ])
    feat_names = feat_names + new_feat_names
    
    # we will model the differences between the prediction target and the most
    # recent observed value
    df['delta_target'] = df['inc_trans_cs_target'] - df['inc_trans_cs']
    
    # if requested, drop features that involve absolute level
    if not incl_level_feats:
        feat_names = _drop_level_feats(feat_names)
    
    return df, feat_names


def _drop_level_feats(feat_names):
    level_feats = ['inc_trans_cs', 'inc_trans_cs_lag1', 'inc_trans_cs_lag2'] + \
                  fnmatch.filter(feat_names, '*taylor_d?_c0*') + \
                  fnmatch.filter(feat_names, '*inc_trans_cs_rollmean*')
    feat_names = [f for f in feat_names if f not in level_feats]
    return feat_names

