from idmodels.preprocess import drop_level_feats

def test_drop_level_feats():
    # a representative subset of the features used by the gbq_qr model
    in_feats = ['inc_trans_cs', 'season_week', 'log_pop', 'source_flusurvnet',
                'source_hhs', 'source_ilinet', 'agg_level_hhs region',
                'agg_level_national', 'agg_level_state', 'location_01',
                'location_02', 'location_04', 'location_05', 'location_06',
                'location_Region 1', 'location_Region 10', 'location_Region 2',
                'location_US', 'delta_xmas', 'inc_trans_cs_taylor_d2_c0_w4t_sNone',
                'inc_trans_cs_taylor_d2_c1_w4t_sNone', 'inc_trans_cs_taylor_d2_c2_w4t_sNone',
                'inc_trans_cs_taylor_d2_c0_w6t_sNone', 'inc_trans_cs_taylor_d2_c1_w6t_sNone',
                'inc_trans_cs_taylor_d2_c2_w6t_sNone', 'inc_trans_cs_taylor_d1_c0_w3t_sNone',
                'inc_trans_cs_taylor_d1_c1_w3t_sNone', 'inc_trans_cs_taylor_d1_c0_w5t_sNone',
                'inc_trans_cs_taylor_d1_c1_w5t_sNone', 'inc_trans_cs_rollmean_w2',
                'inc_trans_cs_rollmean_w4', 'inc_trans_cs_lag1', 'inc_trans_cs_lag2',
                'inc_trans_cs_taylor_d2_c0_w4t_sNone_lag1', 'inc_trans_cs_taylor_d2_c0_w4t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c1_w4t_sNone_lag1', 'inc_trans_cs_taylor_d2_c1_w4t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c2_w4t_sNone_lag1', 'inc_trans_cs_taylor_d2_c2_w4t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c0_w6t_sNone_lag1', 'inc_trans_cs_taylor_d2_c0_w6t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c1_w6t_sNone_lag1', 'inc_trans_cs_taylor_d2_c1_w6t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c2_w6t_sNone_lag1', 'inc_trans_cs_taylor_d2_c2_w6t_sNone_lag2',
                'inc_trans_cs_taylor_d1_c0_w3t_sNone_lag1', 'inc_trans_cs_taylor_d1_c0_w3t_sNone_lag2',
                'inc_trans_cs_taylor_d1_c1_w3t_sNone_lag1', 'inc_trans_cs_taylor_d1_c1_w3t_sNone_lag2',
                'inc_trans_cs_taylor_d1_c0_w5t_sNone_lag1', 'inc_trans_cs_taylor_d1_c0_w5t_sNone_lag2',
                'inc_trans_cs_taylor_d1_c1_w5t_sNone_lag1', 'inc_trans_cs_taylor_d1_c1_w5t_sNone_lag2',
                'inc_trans_cs_rollmean_w2_lag1', 'inc_trans_cs_rollmean_w2_lag2',
                'inc_trans_cs_rollmean_w4_lag1', 'inc_trans_cs_rollmean_w4_lag2', 'horizon']
    
    # subset of in_feats expected to be returned by _drop_level_feats:
    # I have manually removed any that measure local level of the surveillance signal in some way
    # these are 'inc_trans_cs', rolling means of that, and degree 0 coefficients ('c0')
    # of Taylor approximations
    expected = ['season_week', 'log_pop', 'source_flusurvnet',
                'source_hhs', 'source_ilinet', 'agg_level_hhs region',
                'agg_level_national', 'agg_level_state', 'location_01',
                'location_02', 'location_04', 'location_05', 'location_06',
                'location_Region 1', 'location_Region 10', 'location_Region 2',
                'location_US', 'delta_xmas',
                'inc_trans_cs_taylor_d2_c1_w4t_sNone', 'inc_trans_cs_taylor_d2_c2_w4t_sNone',
                'inc_trans_cs_taylor_d2_c1_w6t_sNone',
                'inc_trans_cs_taylor_d2_c2_w6t_sNone',
                'inc_trans_cs_taylor_d1_c1_w3t_sNone',
                'inc_trans_cs_taylor_d1_c1_w5t_sNone',
                'inc_trans_cs_taylor_d2_c1_w4t_sNone_lag1', 'inc_trans_cs_taylor_d2_c1_w4t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c2_w4t_sNone_lag1', 'inc_trans_cs_taylor_d2_c2_w4t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c1_w6t_sNone_lag1', 'inc_trans_cs_taylor_d2_c1_w6t_sNone_lag2',
                'inc_trans_cs_taylor_d2_c2_w6t_sNone_lag1', 'inc_trans_cs_taylor_d2_c2_w6t_sNone_lag2',
                'inc_trans_cs_taylor_d1_c1_w3t_sNone_lag1', 'inc_trans_cs_taylor_d1_c1_w3t_sNone_lag2',
                'inc_trans_cs_taylor_d1_c1_w5t_sNone_lag1', 'inc_trans_cs_taylor_d1_c1_w5t_sNone_lag2',
                'horizon']
    
    actual = drop_level_feats(in_feats)
    
    assert len(actual) == len(expected)
    assert not set(actual) - set(expected)
