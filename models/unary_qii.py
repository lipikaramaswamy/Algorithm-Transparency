#unary_qii.py
#Karina Huang, Lipika Ramaswamy

#This package includes functions used
#to compute unary qii influences.

#import packages
import numpy as np
from collections import defaultdict

#global variables
sex = ['sex_Female', 'sex_Male']
race = ['race_African-American', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
        'race_Native American', 'race_Other']

def intervene(data, var, binary = False):
    '''
    This function intervenes the provided dataframe
    by randomly changing the value of a given qii column.

    Parameters:
    ===========
    1)data: pandas DataFrame object, the original data
    2)var: string, variable to randomize
    3)binary: boolean, whether the variable is binary

    Returns:
    ===========
    dataNew: permuted data frame
    data: the original data
    '''

    #copy data frame to avoid modification
    dataNew = data.copy()

    #if variable is binary
    if binary:
        #randomly flip the bits with probability of 0.5
        dataNew[var] = dataNew[var].apply(lambda x: np.random.binomial(1, 0.5))

        #because we have categorical variables
        #flip the other columns in the category
        if var == sex[0]:
            dataNew[sex[1]] = [1 if x == 0 else 0 for x in dataNew[var].values]
        elif var == sex[1]:
            dataNew[sex[0]] = [1 if x == 0 else 0 for x in dataNew[var].values]
        else:
            varIDX = race.index(var)
            for i in range(len(race)):
                if i != varIDX:
                    dataNew[race[i]] = [1 if x == 0 else 0 for x in dataNew[var].values]

    #if variable is quantitative
    else:
        low = np.min(dataNew[var].values)
        high = np.max(dataNew[var].values)
        #randomly draw a value between the maximum and minimum value in dataset
        dataNew[var] = dataNew[var].apply(lambda x: np.random.randint(low = low, high = high, size = 1)[0])
    return dataNew, data

def computeInfluence(alg, true, permuted, predictors, category):
    '''
    This function computes the unary qii influence for individual outcomes.

    Parameters:
    ===========
    1)alg: trained model for predictions
    2)true: pandas DataFrame, original data
    3)permuted: pandas DataFrame, permuted data
    4)predictors: list or array-like, list of predictors
    5)category: int, decile score categories,
        0 as low, 1 as medium, 2 as high

    Returns:
    ===========
    influence: computed as the absolute difference between
        the predicted probability of being in the category
        if variable is and is not permuted.
    '''
    #prediction with true attributes
    predT = alg.predict_proba(true[predictors])[:, category]
    #prediction with permutated attributes
    predP = alg.predict_proba(permuted[predictors])[:, category]

    influence = np.absolute(np.mean(predT - predP))
    return influence

def individualOutcomes(data, model, predictors):
    '''
    This function computes the unary qii influence for
    individual outcomes for all features in the set of predictors.

    Paramters:
    ==========
    1)data: pandas DataFrame, original data
    2)model: trained model for predictions
    3)predictors: list or array-like, set of features used for prediction

    Returns:
    ==========
    histLow: list, feature influences for low decile scores
    histMed: list, feature influences for medium decile scores
    histHigh: list, feature influences for medium decile scores
    '''

    #cache list for qii influences
    histLow = list()
    histMed = list()
    histHigh = list()

    #binary indicator, hard-coded by order of predictors
    binary = [False, False, False, True, True, True, True, True, True,True,True]

    #for each predictor
    #run 100 experiments and record the average influence
    for i in range(len(predictors)):
        temphistLow = list()
        temphistMed = list()
        temphistHigh = list()
        for n in range(100):
            #permute data
            a, b = intervene(data, predictors[i], binary = binary[i])
            #record influence for each decile score category
            infLow = computeInfluence(model, b, a, predictors, 0)
            infMed = computeInfluence(model, b, a, predictors, 1)
            infHigh = computeInfluence(model, b, a, predictors, 2)
            temphistLow.append(infLow)
            temphistMed.append(infMed)
            temphistHigh.append(infHigh)
        histLow.append(np.mean(temphistLow))
        histMed.append(np.mean(temphistMed))
        histHigh.append(np.mean(temphistHigh))
    return histLow, histMed, histHigh

def computeDiscrepancy(data, group, category):
    '''
    This function computes the discrepancy of being assigned
    a category decile score for for being and not being in the given group.

    Paramters:
    ==========
    1)data: pandas DataFrame object, dataset
    2)group: string, reference group for comparison
    3)category: string, decile score category

    Returns:
    ==========
    discrepancy: absolute difference in probability of being in the category
        between group and non-group individuals
    '''

    #subset groups
    inGroup = data[data[group] == 1]
    outGroup = data[data[group] != 1]

    #probabilities of the category being predicted
    pInGroup = inGroup[inGroup['pred'] == category].shape[0]/inGroup.shape[0]
    pOutGroup = outGroup[outGroup['pred'] == category].shape[0]/outGroup.shape[0]

    #difference in probabilities in vs. out of group
    discrepancy = np.absolute(pInGroup - pOutGroup)

    return discrepancy

def groupDisparity(alg, data, group, reference_group, predictors):
    '''
    This function computes the unary qii for group
    disparity for a given group and set of features.

    Paramters:
    ==========
    1)alg: trained model for predictions
    2)data: pandas DataFrame object, original dataset
    3)group: string, should be 'sex' or 'race' since these
        are the groups included in the current set of predictors
    4)reference_group: string, reference group for comparison
    5)predictors: list or array-like, set of features for predictions

    Returns:
    ==========
    lowAvg: list, group disparity for each feature for low decile scores
    medAvg: list, group disparity for each feature for medium decile scores
    highAvg: list, group disparity for each feature for high decile scores
    rPredictors: list, list of predictors excluding the group variables
    '''

    #copy predictors to avoid modification
    rPredictors = predictors.copy()

    #cache dictionaries for simulations
    histLow = defaultdict(list)
    histMedium = defaultdict(list)
    histHigh = defaultdict(list)

    #for each predictor
    #run 100 experiments and record group disparity 
    if group == 'sex':
        for i in range(len(sex)):
            rPredictors.remove(sex[i])

        for it in range(100):
            for p in rPredictors:

                if p in race:
                    permuted, true = intervene(data, p, binary = True)
                else:
                    permuted, true = intervene(data, p, binary = False)

                #get predictions
                permuted['pred'] = alg.predict(permuted[predictors])
                true['pred'] = alg.predict(true[predictors])

                for c in ['Low', 'Medium', 'High']:
                    influence = computeDiscrepancy(true, reference_group, c) - computeDiscrepancy(permuted, reference_group, c)
                    if c == 'Low':
                        histLow[p].append(influence)
                    elif c == 'Medium':
                        histMedium[p].append(influence)
                    else:
                        histHigh[p].append(influence)

    elif group == 'race':
        for i in range(len(race)):
            rPredictors.remove(race[i])

        for it in range(100):
            for p in rPredictors:

                if p in sex:
                    permuted, true = intervene(data, p, binary = True)
                else:
                    permuted, true = intervene(data, p, binary = False)

                #get predictions
                permuted['pred'] = alg.predict(permuted[predictors])
                true['pred'] = alg.predict(true[predictors])

                for c in ['Low', 'Medium', 'High']:
                    influence = computeDiscrepancy(true, reference_group, c) - computeDiscrepancy(permuted, reference_group, c)
                    if c == 'Low':
                        histLow[p].append(influence)
                    elif c == 'Medium':
                        histMedium[p].append(influence)
                    else:
                        histHigh[p].append(influence)

    #post-processing: record average group disparity
    lowAvg = list()
    medAvg = list()
    highAvg = list()

    for k in rPredictors:
        lowAvg.append(np.mean(histLow[k]))
        medAvg.append(np.mean(histMedium[k]))
        highAvg.append(np.mean(histHigh[k]))

    return lowAvg, medAvg, highAvg, rPredictors
