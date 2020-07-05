import sqlite3
import pandas as pd
import vtreat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
import skopt
import pickle
from opioid_functions import *
import os

os.chdir('/Users/zach.olivier/Desktop/GTX/CSE_6242/course_project')

# define the file
sqlite_file = 'DVADB/DVADB.db'

# open a connection
conn = sqlite3.connect(sqlite_file)

# read from the main table
df = pd.read_sql_query(
    "SELECT * FROM npi_summary",
    conn
)

# take a sample for analysis / modeling
df_model = df.sample(frac=.3)

# quick summary
print(f'dataframe dimensions: {df_model.shape}')
print(f'column names: {df_model.columns}')
print(f' column types: {df_model.dtypes}')

# set index to npi
df_model = df_model.set_index('npi')

# columns that we cannot use for modeling
drop_cols = [
    'nppes_provider_last_org_name',
    'nppes_provider_first_name',
    'nppes_provider_street1',
    'nppes_provider_street2',
    'opioid_claim_count' # does this feature add leakage?
]

df_model.drop(drop_cols, axis=1, inplace=True)

# format columns to numeric
for i in df_model.columns:
    if df_model[i].dtype == object:
        df_model[i] = df_model[i].apply(
            lambda x: 0 if x == '' else x
        )

# clean credentials into usable meta groups
df_model = clean_credentials(df_model)
df_model.drop('nppes_credentials', axis=1, inplace=True)


# list of categorical columns
cols_to_dummy = [
    'nppes_provider_gender', 
    'nppes_provider_city',
    'nppes_provider_state',
    'nppes_provider_country',
    'specialty_description'
]

# dummy out the categorical columns
df_model = dummy_wrapper(df_model, cols_to_dummy)

# TODO: add more features here
# ratio of males to females
# zip code parsing
# state parsing
# combination of specialty and school level
# regionize states

# variance filter
df_model = variance_threshold(df_model, threshold=.01)


## EDA 

# distribution of target
plt.style.use(['dark_background'])
plt.rcParams['figure.figsize'] = (14, 10)
sns.kdeplot(df_model['opioid_prescriber_rate'])
plt.show()

# TODO: box plots of categorical to target

# scatterplot risk score to rate
sns.scatterplot(
    'beneficiary_average_risk_score',
    'opioid_prescriber_rate',
    data=df_model.sample(frac=.01)
).set_title('Opioid Prescriber Rate vs. Risk Score')
plt.show()

# scatterplot total claim to rate
sns.scatterplot(
    np.log(df_model['total_claim_count']),
    'opioid_prescriber_rate',
    data=df_model.sample(frac=.01)
).set_title('Opioid Prescriber Rate vs. Total Claim Count')
plt.show()

# scatterplot avg age to rate
sns.scatterplot(
    'average_age_of_beneficiaries',
    'opioid_prescriber_rate',
    data=df_model.sample(frac=.01)
).set_title('Opioid Prescriber Rate vs. Average Beneficiary Age')
plt.show()

# correlation matrix - spearman
corr = pd.DataFrame(df_model.corr(method='spearman'))

# print heatmap of correlation matrix all features
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cbar=False, cmap='YlGnBu').set_title(
    'Correlation Matrix: Full Feature Set'
)
plt.show()

# correlation matrix - key features - spearman
corr_subset = pd.DataFrame(
    df_model.loc[:, 'total_claim_count':'beneficiary_average_risk_score']
    .corr(method='spearman')
)

# print heatmap of correlation matrix select features
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_subset, cbar=False, cmap='YlGnBu', annot=True
    ).set_title('Correlation Matrix: Select Feature Set')
plt.show()


## MODEL

# divide data into target and predictors
model_target = 'opioid_prescriber_rate'
op_target = df_model[model_target].values
op_features = df_model.drop(model_target, axis=1)

# set up test and train
x_train, x_test, y_train, y_test = train_test_split(
    op_features,
    op_target,
    test_size=.3,
    shuffle=True,
    random_state=789651244
)

# initialize the gbm 
op_gbm = GradientBoostingRegressor(
    learning_rate=0.1,
    loss='lad',
    max_depth=200,
    max_features='sqrt',
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    min_samples_leaf=.05,
    min_samples_split=.05,
    n_estimators=200,
    presort='auto',
    random_state=758491629,
    subsample=0.3,
    verbose=10,
    warm_start=False
)

# fit the model
op_gbm_fit = op_gbm.fit(x_train, y_train)

# cross validate the gbm model
cross_validate(
    op_gbm_fit,
    X=x_train,
    y=y_train,
    cv=10,
    scoring='neg_mean_absolute_error'
)

# extract variable importance for the gbm model
df_vimp, vimp_columns = extract_vimp(
    op_gbm_fit,
    column_names=x_train,
    threshold=.004
)

plt.rcParams['figure.figsize'] = (14, 10)
sns.barplot(
    x='tree_vimp',
    y='variable',
    data=df_vimp
).set_title('Variable Importance: Opioid Rate Model')

plt.show()


# set up bayesian grid search
opt = skopt.BayesSearchCV(
    op_gbm,
    {
        "learning_rate": (1e-8, .98, 'log-uniform'),
        "max_depth": (1, 300),
        "n_estimators": (20, 200),
        "max_features": ["sqrt", "log2"],
        "subsample": (0.3, .9, 'log-uniform'),
        "min_samples_split": (.005, .1),
        "min_samples_leaf": (.005, .1)
    },
    n_iter=40,
    cv=5,
    scoring='neg_mean_absolute_error'
)

# fit parameters
op_opt_gbm = opt.fit(x_train, y_train)

# cross validate the bayesian optimized gbm model
cross_validate(
    op_opt_gbm,
    X=x_train,
    y=y_train,
    cv=10
)

# show the best params
print("val. score: %s" % opt.best_score_)
print(opt.best_estimator_)

# initialize the gbm regressor
op_gbm_bayes = GradientBoostingRegressor(
    alpha=0.9, criterion='friedman_mse', init=None,
    learning_rate=0.15066211175185815, loss='lad', max_depth=203,
    max_features='log2', max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=0.005, min_samples_split=0.005,
    min_weight_fraction_leaf=0.0, n_estimators=200,
    presort='auto', random_state=758491629, subsample=0.9,
    verbose=10, warm_start=False
)

# fit the regressor
op_gbm_bayes_fit = op_gbm_bayes.fit(x_train, y_train)

# cross validate the gbm model
cross_validate(
    op_gbm_bayes_fit,
    X=x_train,
    y=y_train,
    cv=10,
    scoring='neg_mean_absolute_error'
)

# extract variable importance for the gbm model
df_vimp, vimp_columns = extract_vimp(
    op_gbm_bayes_fit,
    column_names=x_train,
    threshold=.004
)

# plot variable importance
plt.rcParams['figure.figsize'] = (14, 10)
sns.barplot(
    x='tree_vimp',
    y='variable',
    data=df_vimp
).set_title('Variable Importance: Bayesian Gradient Boosting')

plt.show()

# get model train predictions
y_pred = op_gbm_bayes_fit.predict(x_train)
print(f'train predictions mean {np.mean(y_pred)}')
print(f'train predictions median {np.median(y_pred)}')
print(f'train predictions var {np.var(y_pred)}')

# plot predictions distribution
sns.distplot(y_pred).set_title('Train Predictions Distribution')
plt.show()

# plot predictions vs. actuals - train set
sns.scatterplot(
    x=y_pred,
    y=y_train
).set_title('Opioid Rate Predictions vs. Train')
plt.xlabel('predictions')
plt.ylabel('train actuals')
plt.show()

# put train predictions and actuals together
df_comb = pd.concat([pd.Series(y_pred), pd.Series(y_train)], axis=1)
df_comb.columns = ['predictions', 'actuals']

# get difference
df_comb['error'] = np.abs(df_comb['predictions'] - df_comb['actuals'])

# plot predictions vs. actuals - train set
sns.scatterplot(
    x='predictions',
    y='actuals',
    data=df_comb.sample(frac=.3),
    hue='error'
).set_title('Opioid Rate Predictions vs. Train')
plt.xlabel('predictions')
plt.ylabel('train actuals')
plt.show()

# get model test predictions
y_pred_test = op_gbm_bayes_fit.predict(x_test)
print(f'test predictions mean {np.mean(y_pred_test)}')
print(f'test predictions median {np.median(y_pred_test)}')
print(f'test predictions var {np.var(y_pred_test)}')

# plot predictions distribution
sns.distplot(y_pred_test).set_title('Test Predictions Distribution')
plt.show()

# plot predictions vs. actuals - train set
sns.scatterplot(
    x=y_pred_test,
    y=y_test
).set_title('Opioid Rate Predictions vs. Test')
plt.xlabel("Predictions")
plt.ylabel("Actuals")
plt.show()

# put train predictions and actuals together
df_comb = pd.concat([pd.Series(y_pred_test), pd.Series(y_test)], axis=1)
df_comb.columns = ['predictions', 'actuals']

# get difference
df_comb['error'] = np.abs(df_comb['predictions'] - df_comb['actuals'])

# plot predictions vs. actuals - train set
sns.scatterplot(
    x='predictions',
    y='actuals',
    data=df_comb.sample(frac=.3),
    hue='error',
    size='error'
).set_title('Opioid Rate Predictions vs. Test')
plt.xlabel('predictions')
plt.ylabel('train actuals')
plt.show()


# TODO: shap values
# TODO: transformation functions for predicting on entire data set
# TODO: output frames for plotting shap, residuals
# TODO: output final model for predicting on entire dataset

# final model to production
pickle.dump(
    op_gbm_bayes_fit,
    open('opioid_gbm_full.sav', 'wb')
)

# final features to production
pickle.dump(
    x_train.columns,
    open('opioid_features.sav', 'wb')
)

