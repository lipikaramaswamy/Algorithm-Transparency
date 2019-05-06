""" Various QII related computations. """

import pandas as pd
import numpy as np
from builtins import range

def intervene(data, varList, binary):
    """
    Intervene on a set of columns from the data

    Parameters
    ==========
    - data: pandas dataframe
    - varList: set of columns to intervene on, i.e. shuffle such as [x1, x2]
    - binary: list corresponding to varList of set of booleans corresponding to whether the column is one-hot-encoded or not such as [True, False]

    Returns
    =======
    dataframe with selected columns permuted

    """
    dataNew = data.copy()
    sex = ['sex_Female', 'sex_Male']
    race = ['race_African-American', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
            'race_Native American', 'race_Other']

    for var in varList:
        if binary:
            dataNew[var] = dataNew[var].apply(lambda x: np.random.binomial(1, 0.5))
            if var == sex[0]:
                dataNew[sex[1]] = [1 if x == 0 else 0 for x in dataNew[var].values]
            elif var == sex[1]:
                dataNew[sex[0]] = [1 if x == 0 else 0 for x in dataNew[var].values]
            else:
                varIDX = race.index(var)
                for i in range(len(race)):
                    if i != varIDX:
                        dataNew[race[i]] = [1 if x == 0 else 0 for x in dataNew[var].values]
        else:
            low = np.min(dataNew[var].values)
            high = np.max(dataNew[var].values)
            dataNew[var] = dataNew[var].apply(lambda x: np.random.randint(low = low, high = high, size = 1)[0])
    return dataNew



def random_intervene_point(data, varList, index_of_person, binary):
    """ Randomly intervene on a set of columns of x from X. """

    dataNew = data.copy()
    sex = ['sex_Female', 'sex_Male']
    race = ['race_African-American', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
            'race_Native American', 'race_Other']

    n = data.shape[0]
    order = np.random.permutation(range(n))
    dataNew = pd.concat([data.loc[index_of_person, :]] * n, axis = 1).T
    for k, var in enumerate(varList):
        if binary[k]:
            dataNew.loc[:, var] = dataNew.loc[:, var].apply(lambda x: np.random.binomial(1, 0.5))
            if var == sex[0]:
                dataNew[sex[1]] = [1 if x == 0 else 0 for x in dataNew[var].values]
            elif var == sex[1]:
                dataNew[sex[0]] = [1 if x == 0 else 0 for x in dataNew[var].values]
            else:
                varIDX = race.index(var)
                for i in range(len(race)):
                    if i != varIDX:
                        dataNew[race[i]] = [1 if x == 0 else 0 for x in dataNew[var].values]
        else:
            low = np.min(data[var].values)
            high = np.max(data[var].values)
            dataNew[var] = dataNew[var].apply(lambda x: np.random.randint(low = low, high = high, size = 1)[0])
    return dataNew

def unary_individual_influence(dataset, cls, x_ind, varList, binary, predictors):
    y_pred = cls.predict(np.array(dataset.loc[x_ind, predictors]).reshape(1,-1))
    average_local_inf = {}
    iters = 1
    for sf in varList:
        local_influence = np.zeros(y_pred.shape[0])
        for i in range(0, iters):
            X_inter = random_intervene_point(dataset, [sf], x_ind, binary)
            y_pred_inter = cls.predict(np.array(X_inter.loc[:,predictors]))
            local_influence = local_influence + (y_pred == y_pred_inter)*1.
        average_local_inf[sf] = 1 - (local_influence/iters).mean()
    return average_local_inf


def shapley_influence(dataset, cls, x_ind, X_test, varList, predictors):

    p_samples = 600
    s_samples = 600

    def intervene(S_feature, X_values, X_inter):
        for f in S_feature:
            X_values[:, f] = X_inter[:, f]

    def P(X_values):
        preds = (cls.predict(X_values))
        return ((preds == y0) * 1.).mean()

    y0 = cls.predict(np.array(dataset.loc[x_ind, predictors]).reshape(1,-1))
    b = np.random.randint(0, X_test.shape[0], p_samples)
    X_sample = np.array(X_test.iloc[b,:])

    # translate into integer indices
    ls = {}
    for si in varList:
        ls[si] = [X_test.columns.get_loc(f) for f in varList]
    shapley = dict.fromkeys(varList, 0)

    for sample in range(0, s_samples):
        perm = np.random.permutation(len(varList))
        # X_data is x_individual intervened with some features from X_sample
        # Invariant: X_data will be x_individual intervened with Union of si[perm[ 0 ... i-1]]
        X_data = np.array(pd.concat([dataset.loc[x_ind, :]] * p_samples, axis = 1).T)

        # p for X_data, == 1.0 trivially at start.
        p_S_si = 1.0

        for i in range(0, len(varList)):
            # Choose a random subset and get string indices by flattening
            #  excluding si
            si = varList[perm[i]]

            #repeat x_individual_rep
            intervene(ls[si], X_data, X_sample)

            p_S = P(X_data)
            shapley[si] = shapley[si] - (p_S - p_S_si)/s_samples
            p_S_si = p_S

    return shapley
