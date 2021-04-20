import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def load(drop_outliers=True, describe=True):
    #loading data to the datafrmae
    df = pd.read_excel('initial_data.xlsx')
    #creating subset of nescessary columns [ Unitprice Pal grossweight  Pal height  Units per pal]
    df = df[['Pal grossweight', 'Pal height', 'Units per pal', 'Unitprice']]
    #print(len(df.index), " originally")
    df = df[(df[['Pal grossweight', 'Pal height', 'Units per pal', 'Unitprice']] != 0).all(axis=1)]
    if drop_outliers == True:
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    if describe==True:
        print(df.describe())
        print('Median: ', df.median())
    print(len(df.index), " observations remained")
    return df

def heatmap(data):
    corr = data.corr()
    print(corr)
    # plot the heatmap
    #sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    ax = sns.heatmap(corr, annot=True)
    plt.show()

def predictions(data):

    df = data
    #df = shuffle(data,  random_state=10)
    df = shuffle(data, random_state=10)

    x = df[['Pal grossweight', 'Pal height', 'Units per pal']].values
    y = df['Unitprice'].values

    reg = LinearRegression().fit(x, y)
    scores_reg = cross_validate(reg, x, y, cv=20, scoring = ('r2', 'neg_mean_squared_error'))
    #print(np.mean(scores_reg['test_neg_mean_squared_error']))
    #print(np.mean(scores_reg['test_r2']))

    reg = KNeighborsRegressor(n_neighbors=6, weights='uniform', leaf_size=30, p=2, metric='minkowski')
    scores_knn = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))

    reg = DecisionTreeRegressor()
    scores_tree = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))

    reg = make_pipeline(StandardScaler(), SVR())
    scores_svr = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))

    reg = MLPRegressor(max_iter=300)
    scores_mlp = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))


    reg = RandomForestRegressor(n_estimators=100)
    scores_forest = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_root_mean_squared_error'))

    sns.set_theme(style="whitegrid")

    stats = pd.DataFrame(list(zip(scores_reg['test_r2'], scores_knn['test_r2'], scores_tree['test_r2'],
                                  scores_svr['test_r2'], scores_mlp['test_r2'], scores_forest['test_r2'],)),
                         columns=['OLS', 'KNN', 'Tree', 'SVR', 'MLP', 'Forest'])

    sns.boxplot(data=stats)
    plt.ylabel('R-squared')
    plt.show()

def hp_opt_svm(data):

    df = data
    #df = shuffle(data,  random_state=10)
    df = shuffle(data, random_state=10)

    x = df[['Pal grossweight', 'Pal height', 'Units per pal']].values
    y = df['Unitprice'].values

    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
    reg = make_pipeline(StandardScaler(), SVR(kernel='linear'))
    scores_svr_lin = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))

    reg = make_pipeline(StandardScaler(), SVR(kernel='poly', degree=3))
    scores_svr_poly = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))

    reg = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    scores_svr_rbf = cross_validate(reg, x, y, cv=20, scoring=('r2', 'neg_mean_squared_error'))

    sns.set_theme(style="whitegrid")

    stats = pd.DataFrame(list(zip(scores_svr_lin['test_r2'], scores_svr_poly['test_r2'], scores_svr_rbf['test_r2'])),
                         columns=['Linear', 'Poly', 'RBF'])

    sns.boxplot(data=stats)
    plt.ylabel('R-squared')
    plt.show()


def hp_opt_knn(data):

    df = data
    #df = shuffle(data,  random_state=10)
    df = shuffle(data, random_state=10)

    x = df[['Pal grossweight', 'Pal height', 'Units per pal']].values
    y = df['Unitprice'].values

    stats = []
    Ks = [i for i in range(1, 60)]
    for i in Ks:
        reg = KNeighborsRegressor(n_neighbors=i, weights='uniform', leaf_size=30, p=2, metric='minkowski')
        score = cross_validate(reg, x, y, cv=4, scoring=('r2', 'neg_mean_squared_error'))
        stats.append(score['test_r2'])
    means, stds = [], []
    for i in stats:
        means.append(np.mean(i))
        stds.append(np.std(i))

    plt.plot(Ks, means, label="Mean cross-validation score", color="navy")
    plt.fill_between(Ks, list(np.array(means) - np.array(stds)), list(np.array(means) + np.array(stds)), label="Standart diviation", alpha=0.2,
                     color="navy")
    plt.grid()
    plt.legend()
    plt.xlabel('Number of neighbours (k)')
    plt.ylabel('R-squared')
    plt.show()


def pairplot(data):
    '''sns.pairplot(data)
    plt.show()'''
    g = sns.pairplot(data, diag_kind="kde")
    plt.show()

if __name__ == '__main__':
    df = load()
    #hp_opt_svm(df)
    #predictions(df)
    #heatmap(df)
    #pairplot(df)
    hp_opt_knn(df)