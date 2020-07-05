from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn import tree
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
# from sklearn.tree._tree import TREE_LEAF
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

def clean_credentials(df):

    # MD
    df['credential_md'] = np.where(
        df['nppes_credentials'].str.replace(".", "").str.contains("MD"),
        1,
        0
    )

    # RN
    df['credential_rn'] = np.where(
        df['nppes_credentials'].str.replace(".", "").str.contains("RN"),
        1,
        0
    )

    # PHD
    df['credential_phd'] = np.where(
        df['nppes_credentials'].str.replace(".", "").str.contains("PHD"),
        1,
        0
    )

    # DDS
    df['credential_dds'] = np.where(
        df['nppes_credentials'].str.replace(".", "").str.contains("DDS"),
        1,
        0
    )

    # PA
    df['credential_pa'] = np.where(
        df['nppes_credentials'].str.replace(".", "").str.contains("PA"),
        1,
        0
    )

    # MBA
    df['credential_mba'] = np.where(
        df['nppes_credentials'].str.replace(".", "").str.contains("MBA"),
        1,
        0
    )

    return df

def ks_distribution_test(df_a, df_b, alpha=.05):
    dist_test = stats.ks_2samp(df_a, df_b)

    if dist_test[1] < alpha:
        print(
            df_a.name,
            ': retained and cancelled distributions are different'
        )

    else:
        print(
            df_b.name,
            ': retained and cancelled distributions are the same'
        )


def dummy_wrapper(df, cols_to_dummy=None):
    """
    Wrapper for pd.get_dummies that appends dummy variables back onto original dataset, cleans columns
    Parameters
    ----------
    df : DataFrame
    cols_to_dummy : list
    Returns
    -------
    DataFrame
    """

    df = df.copy()

    df_dummy = pd.get_dummies(df[cols_to_dummy], dummy_na=True)

    # clean the categorical column names
    df_dummy.columns = df_dummy.columns. \
        str.strip(). \
        str.lower(). \
        str.replace(' ', '_'). \
        str.replace('-', ''). \
        str.replace('/', ''). \
        str.replace('$', ''). \
        str.replace(',', ''). \
        str.replace('&', ''). \
        str.replace('.', ''). \
        str.replace('+', ''). \
        str.replace(':', ''). \
        str.replace('|', ''). \
        str.replace('[', ''). \
        str.replace(']', ''). \
        str.replace('(', '').str.replace(')', '')

    cols_to_keep = [c for c in df.columns if c not in cols_to_dummy]

    df_keep = df[cols_to_keep]

    df_clean = pd.concat([df_keep, df_dummy], axis=1)

    return df_clean


def variance_threshold(df, threshold=.001):
    """
    Checks model frame for numeric columns with zero variance and variance
    less than a provided threshold
    Parameters
    ----------
    df : DataFrame
    threshold: float
    Returns
    -------
    DataFrame
    """

    print(f"number of features before filter {df.shape[1]}")

    var_dict = np.var(df, axis=0).to_dict()

    var_dict = {k: v for (k, v) in var_dict.items() if v >= threshold}

    keep_list = list(var_dict.keys())

    df = df[keep_list]

    print(f"number of features remaining {df.shape[1]}")

    return df


def clean_categorical(df):
    """
    Simple parser to exclude random categorical column characters from the final value
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    DataFrame
    """

    df = df.str.strip(). \
        str.lower(). \
        str.replace(' ', '_'). \
        str.replace('-', ''). \
        str.replace('/', ''). \
        str.replace('$', ''). \
        str.replace(',', ''). \
        str.replace('&', ''). \
        str.replace('.', ''). \
        str.replace('|', ''). \
        str.replace('(', '').str.replace(')', '')

    return df


def extra_trees_vimp(
        df,
        y,
        threshold=.01,
        plot=True,
        estimators=100,
        depth=3,
        split_sample=.05,
        leaf_sample=.05,
        transform=True
):

    print('Building Trees...')

    x_vars = df
    y_vars = y

    # flow control for regression or classification
    regression_type = y_vars.drop_duplicates()

    if len(regression_type) == 2:

        print('Building Classification Trees...')

        model = ExtraTreesClassifier(
            n_estimators=estimators,
            max_depth=depth,
            random_state=444,
            min_samples_split=split_sample,
            min_samples_leaf=leaf_sample,
            class_weight='balanced_subsample',
            max_features='log2',
            bootstrap=True,
            oob_score=True
            )

        model.fit(x_vars, np.asarray(y_vars).ravel())

        importance = model.feature_importances_

        df = pd.DataFrame(importance)
        df = df.T
        df.columns = x_vars.columns
        df = df.T.reset_index()
        df.columns = ['variable', 'tree_vimp']
        df = df.sort_values('tree_vimp', ascending=False)

    else:

        if transform:
            y_vars = np.sqrt(y_vars)

        print('Building Regression Trees...')

        model = ExtraTreesRegressor(
            n_estimators=estimators,
            max_depth=depth,
            random_state=444,
            min_samples_split=split_sample,
            min_samples_leaf=leaf_sample,
            max_features='log2',
            bootstrap=True,
            oob_score=True
            )
        model.fit(x_vars, np.asarray(y_vars).ravel())

        importance = model.feature_importances_

        df = pd.DataFrame(importance)
        df = df.T
        df.columns = x_vars.columns
        df = df.T.reset_index()
        df.columns = ['variable', 'tree_vimp']
        df = df.sort_values('tree_vimp', ascending=False)

    if plot:
        plt.figure()
        sns.barplot(
            x='tree_vimp',
            y='variable',
            data=df[df.tree_vimp >= threshold],
            palette='Blues_r',
        ).set_title(y.name)

    # extract the best tree importance results
    df = df[df.tree_vimp >= threshold]
    important_cols = list(df.variable)

    print('Tree Models Complete')

    return df, important_cols, model.oob_score_


def clean_multi_index_headers(df):
    """
    Concatenates a multi-index columns headers into one with a clean format
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    DataFrame
    """

    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    return df


# set up the plot decision tree function
def plot_cart(
        estimator,
        file_name='my_tree.dot',
        feature_name='metric1',
        class_name=('LOST', 'WON')
):
    """
    Visualizes a CART decision tree using graphviz plotting
    Parameters
    ----------
    estimator : fitted decision tree object
    file_name : filename to store the graphviz plot
    feature_name : name of features available in decision tree
    class_name : label of the classifier "classes" the decision tree solves for
    Returns
    -------
    graphviz object
    """

    tree.export_graphviz(
        estimator,
        out_file=file_name,
        filled=True,
        special_characters=True,
        rounded=True,
        feature_names=feature_name,
        class_names=class_name,
        proportion=True,
        rotate=False,
        precision=3
    )

    with open(file_name) as f:
        dot_graph = f.read()

    return graphviz.Source(dot_graph)


def extract_vimp(clf, column_names, threshold=.001):
    """
    Helper function to extract variable importance from the feature importance based model
    Takes in a fit object and extract model feature importance into a dataframe
    Need to supply column headers
    Parameters
    ----------
    clf : model.fit() object
    column_names : dataframe of features
    threshold : limit returned values to by vimp threshold
    Returns
    -------
    DataFrame
    """

    importance = clf.feature_importances_

    df = pd.DataFrame(importance)
    df = df.T
    df.columns = column_names.columns
    df = df.T.reset_index()
    df.columns = ['variable', 'tree_vimp']
    df = df.sort_values('tree_vimp', ascending=False)

    # extract the best tree importance results
    df = df[df.tree_vimp >= threshold]
    important_cols = list(df.variable)

    return df, important_cols


def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "un linking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF

    # if there are children, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


def get_difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return pd.Series(diff)


def tree_weight_cv(target, features, weight_max=20):
    """
    Quick cross validation for class weights for a decision tree classifier
    Parameters
    ----------
    target : target values to fit as a classifier
    features : frame of features to fit in model
    weight_max : range max to cross validate through
    Returns
    -------
    set
    """

    weight_dict = dict()

    for i in range(0, weight_max, 1):
        print(i)

        # initialize the CART
        mvc_tree = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_leaf_nodes=300,
            random_state=478946,
            # class_weight='balanced',
            class_weight={1: i, 0: 1},
            min_samples_leaf=.0001,
            min_samples_split=.0001
        )

        # fit the decision tree
        estimator: object = mvc_tree.fit(
            X=features,
            y=target
        )

        # get the cross validation score for fitted model
        cv_tree = cross_validate(
            estimator,
            features,
            target,
            cv=10,
            scoring='roc_auc',
            return_estimator=True,
            return_train_score=True
        )

        print(np.mean(cv_tree['test_score']))
        print(np.mean(estimator.predict(features)))

        weight_dict[i] = np.mean(cv_tree['test_score'])

    return {weight_dict, estimator, cv_tree}


def scale_variables(train, test):
    """
    Applies StandardScalar() learned on the training dataset to the training and test data frames
    Applying the learned scalar on train to the test frames helps avoid information leakage
    Parameters
    ----------
    train : training data frame of features
    test : testing data frame of features
    Returns
    -------
    DataFrame : scaled training and test set data frames
    fit object: StandardScaler() fit object to use for future predictions
    """
    selector = StandardScaler()

    return (
        pd.DataFrame(selector.fit_transform(train), columns=train.columns, index=train.index),
        pd.DataFrame(selector.transform(test), columns=train.columns),
        selector.fit(train)
    )


def feature_contributions(model, df):
    """
    Extract the shapley contributions for a tree based model
    Parameters
    ----------
    model : fitted tree based model from sklearn
    df : frame of training or test data
    Returns
    -------
    set
    """

    explain_tree = shap.TreeExplainer(model)
    shp_values = explain_tree.shap_values(df)

    plt.figure(figsize=(10, 20))
    shap.summary_plot(shp_values, df)
    plt.show()

    return pd.DataFrame(np.concatenate(shp_values), columns=df.columns), explain_tree, shp_values


def shp_outputs(model, x_shp_, y_shp_, plot_results=True):
    """
    Extract the shapley contributions for a tree based model
    Save features in a full data frame
    Plot features in barplot for global contribution by feature
    Plot features in a heatmap for local contribution by training / test sample
    Parameters
    ----------
    model : fitted tree based model from sklearn
    x_shp_ : frame of training or testing observations
    y_shp_ : frame of training or testing responses
    plot_results : indicator to plot the barplot and heatmap with seaborn
    Returns
    -------
    set
    """
    fc_, shp_explain, shp_values = feature_contributions(model, x_shp_)
    fc_ = pd.concat(
        [fc_.reset_index(), pd.Series(y_shp_).rename('index')],
        axis=1,
        ignore_index=False
    ).drop('index', axis=1)

    # group all features by their average shp contribution
    fc_summary = pd.DataFrame(fc_.mean()).reset_index()
    fc_summary.columns = ['feature', 'avg_contribution']
    fc_summary['contribution_type'] = np.where(
        fc_summary.avg_contribution < 0,
        'negative',
        'positive'
    )

    if plot_results:
        plt.rcParams['figure.figsize'] = (14, 16)
        sns.barplot(
            x='avg_contribution',
            y='feature',
            data=fc_summary[abs(fc_summary.avg_contribution) > 0.0],
            hue='contribution_type',
            palette=['#FF0D57', '#1E88E5']
        ).set_title('Average Shapley Contribution by Feature')
        plt.figure()
        plt.show()

        # subset out random columns
        keep_cols = fc_.iloc[:, :-1].columns.to_series().sample(frac=.20)

        plt.rcParams['figure.figsize'] = (14, 6)
        sns.heatmap(
            fc_[keep_cols].sample(50).T,
            center=np.mean(fc_.values),
            robust=True,
            cbar=False,
            square=True,
            xticklabels=False,
            cmap=['#FF0D57', '#FFC3D5', 'whitesmoke', '#1E88E5', '#D1E6FA']
        ).set_title('Variable Contribution: Churn Model')
        plt.figure()
        plt.show()

    return fc_, shp_explain, shp_values, fc_summary


def feature_extraction_forest(
        df=None,
        features=None,
        target=None,
        folds=5,
        step_size=.01,
        model_type='classifier',
        select_n=2
):
    """
    Recursive Feature Extraction wrapper
    Run RFE with model type "classifier" or "regression" depending on target
    Return the top features selected from the RFE process
    Parameters
    ----------
    df : data frame of training and test responses
    features : frame of features
    target : frame of targets
    folds : number of cross validation folds for the RFE estimator to be tested against
    step_size : percent of features to eliminate at each round, based on variable importance
    model_type : indicator letting process know we are in a classification or regression problem
    select_n : number of ranked features to keep, 1 indicates the best ranked features set
    Returns
    -------
    set
    """

    if model_type == 'classifier':

        # initialize the random forest for feature extraction
        rf_estimator = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=.03,
            min_samples_split=.03,
            max_features='log2',
            bootstrap=True
        )

        rf_fit = rf_estimator.fit(X=features, y=target)
        tree_vars = extract_vimp(rf_fit, column_names=features)

        # set up the automated feature extraction model
        rfe = RFECV(
            estimator=rf_estimator,
            cv=folds,
            step=step_size,
            verbose=3
        )

        # fit the RFE model
        rfe_fit = rfe.fit(X=features, y=target)

        # rfe details
        ranked_features = list(rfe_fit.ranking_)
        best_features = [i for i, f in enumerate(ranked_features) if f <= select_n]
        best_cols = list(df.iloc[:, best_features].columns)
        best_cols = list(set(best_cols))
        print(len(best_cols))

        return best_cols, best_features, tree_vars

    else:

        # initialize the random forest for feature extraction
        rf_estimator = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=.03,
            min_samples_split=.03,
            max_features='log2',
            bootstrap=True
        )

        rf_fit = rf_estimator.fit(X=features, y=target)
        tree_vars = extract_vimp(rf_fit, column_names=features)

        # set up the automated feature extraction model
        rfe = RFECV(
            estimator=rf_estimator,
            cv=folds,
            step=step_size,
            verbose=3
        )

        # fit the RFE model
        rfe_fit = rfe.fit(X=features, y=target)

        # rfe details
        ranked_features = list(rfe_fit.ranking_)
        best_features = [i for i, f in enumerate(ranked_features) if f <= select_n]
        best_cols = list(df.iloc[:, best_features].columns)
        best_cols = list(set(best_cols))
        print(len(best_cols))

        return best_cols, best_features, tree_vars


def plot_performance(gcv):
    n_splits = gcv.cv.n_splits
    cv_scores = {"alpha": [], "test_score": [], "split": []}
    order = []
    for i, params in enumerate(gcv.cv_results_["params"]):
        name = "%.5f" % params["alpha"]
        order.append(name)
        for j in range(n_splits):
            vs = gcv.cv_results_["split%d_test_score" % j][i]
            cv_scores["alpha"].append(name)
            cv_scores["test_score"].append(vs)
            cv_scores["split"].append(j)
    df = pd.DataFrame.from_dict(cv_scores)
    _, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(x="alpha", y="test_score", data=df, order=order, ax=ax)
    _, x_text = plt.xticks()
    for t in x_text:
        t.set_rotation("vertical")
        