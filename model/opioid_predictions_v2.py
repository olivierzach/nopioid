import sqlite3
import pandas as pd
import pickle
from opioid_functions import *
import os

os.chdir('/Users/zach.olivier/Desktop/GTX/CSE_6242/course_project')

# import the final features from development model
features = list(pd.read_pickle('opioid_features.sav'))

# load the model
loaded_model = pickle.load(open('opioid_gbm_full.sav', 'rb'))

# define the file
sqlite_file = 'DVADB/DVADB.db'

# open a connection
conn = sqlite3.connect(sqlite_file)

# read from the main table
df = pd.read_sql_query(
    "SELECT * FROM npi_summary WHERE nppes_provider_country = 'US'",
    conn
)

# tuck in npi to expose for joining later
df = df.set_index('npi')

# columns that we cannot use for modeling
drop_cols = [
    'nppes_provider_last_org_name',
    'nppes_provider_first_name',
    'nppes_provider_street1',
    'nppes_provider_street2',
    'opioid_claim_count' # does this feature add leakage?
]

df.drop(drop_cols, axis=1, inplace=True)

# format columns to numeric
for i in df.columns:
    if df[i].dtype == object:
        df[i] = df[i].apply(
            lambda x: 0 if x == '' else x
        )

# clean credentials into usable meta groups
df = clean_credentials(df)
df.drop('nppes_credentials', axis=1, inplace=True)

# do not need to dummy every individual city - only the cities the model needs for prediction
df['nppes_provider_city'] = np.where(
    df['nppes_provider_city'] == 'HOUSTON',
    df['nppes_provider_city'],
    np.where(
        df['nppes_provider_city'] == 'NEW YORK',
        df['nppes_provider_city'],
        np.where(
            df['nppes_provider_city'] == 'CHICAGO',
            df['nppes_provider_city'],
            'Other'
        )
    )  
)

# list of categorical columns
cols_to_dummy = [
    'nppes_provider_gender', 
    'nppes_provider_city',
    'nppes_provider_state',
    'nppes_provider_country',
    'specialty_description'
]

# dummy out the categorical columns
df = dummy_wrapper(df, cols_to_dummy)

# drop target from prediction set
df.drop(
    'opioid_prescriber_rate',
    axis=1,
    inplace=True
)

# subset to only the needed features
df = df[features]

# some duplicated columns leak in...
df = df.loc[:, ~df.columns.duplicated()]

# predict onto the entire dataset 
gbm_predictions = loaded_model.predict(df)

# append predictions onto original frame
df['gbm_predict'] = gbm_predictions

# extract the shap values
shp_df, explain, shp_v = feature_contributions(loaded_model, df)

# append shp values back onto original prediction frame
df = df.merge(
    shp_df,
    how='inner',
    left_index=True,
    right_index=True,
    suffixes=('', '_shp')
)

# expose npi for joining
df = df.reset_index()

# write final results back into sqlite database
if conn is not None:
    df.to_sql(
        'npi_predictions',
        conn,
        if_exists='replace'
    )

# read in original for joining
df_original = pd.read_sql_query(
    "SELECT * FROM npi_summary WHERE nppes_provider_country = 'US'",
    conn
)

# combine predictions with original
df_combined = df_original.merge(
    df,
    how='inner',
    on='npi'
)

# calculate residuals
df_combined['diff'] = df_combined['opioid_prescriber_rate'] - df_combined['gbm_predict']
df_combined['abs_diff'] = abs(df_combined['diff'])

# write final results back into sqlite database
if conn is not None:
    df_combined.to_sql(
        'npi_summary_with_predictions',
        conn,
        if_exists='replace'
    )

# subset to only large residuals
df_outliers = df_combined[df_combined['abs_diff'] >= 20]
df_outliers.shape


# write final results back into sqlite database
if conn is not None:
    df_outliers.to_sql(
        'npi_outliers_only',
        conn,
        if_exists='replace'
    )
