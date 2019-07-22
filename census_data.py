import pandas as pd
import censusdata
import collections

pd.set_option('display.expand_frame_repr', False)

    ## B19037 - AGE OF HOUSEHOLDER BY HOUSEHOLD INCOME
        ## B11005 - HOUSEHOLDS BY PRESENCE OF PEOPLE UNDER 18 YEARS
        ## B11007 - HOUSEHOLDS BY PRESENCE OF PEOPLE OVER 65 YEARS
        ## B25011 - TENURE BY HOUSEHOLD TYPE/AGE (incl. living alone) - HOUSEHOLD TYPE.
    ## B25118 - TENURE BY HOUSEHOLD INCOME

## age x income x tenure - household type x children x elders
## this will give - age, children, elders, married, alone, income. don't need gender.

# i don't think these are relevant but you never know:
## B11012 - HOUSEHOLD TYPE BY TENURE: RENT/OWN
## B25003 - OCCUPIED HOUSING UNITS - OWNED/RENTED
## B11001 - HOUSEHOLD TYPE (INCL. LIVING ALONE) -- only relevant for living alone
## B25115 - TENURE BY HOUSEHOLD TYPE, AGE, CHILDREN

## can use below code as a framework for implementing household conversion

def createShells(tableCode, startShell, endShell):
    tableCode += '_0'
    shellList = []
    for i in range(startShell, endShell+1):
        shell = tableCode
        if i < 10:
            shell += str(0)
        shell += str(i)
        shell += 'E'
        shellList.append(shell)
    return shellList

"""
shells = createShells('B11005', 1, 19)
mult = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '41'), ('county', '051')]), shells)
la = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '06'), ('county', '037')]), shells)
bos = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '25'), ('county', '025')]), shells)
dc = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '11'), ('county', '001')]), shells)
testData = la.append(dc)
testData = testData.append(bos)
testData = testData.append(mult)
testData.to_csv('raw_household65a.csv')
"""

def csvToDict(filepath, name):
    ageData = pd.read_csv(filepath)
    cumAgeData = ageData.cumsum(axis=1, skipna=True)
    transAgeData = cumAgeData.transpose()
    ageDataDict = transAgeData.to_dict('list')
    ageData_toMerge = collections.defaultdict(dict)
    for i in range(4):
        ageData_toMerge[i][name] = ageDataDict[i]
    return ageData_toMerge

def csvToDict_noCumSum(filepath, name):
    ageData = pd.read_csv(filepath)
    transAgeData = ageData.transpose()
    ageDataDict = transAgeData.to_dict('list')
    ageData_toMerge = collections.defaultdict(dict)
    for i in range(4):
        ageData_toMerge[i][name] = ageDataDict[i]
    return ageData_toMerge

def mergeDicts():
    ageData = csvToDict('test_data/pcthouseholdagedist.csv', 'age')
    u25income = csvToDict('test_data/pctincomeageu25.csv', 'u25income')
    income2544 = csvToDict('test_data/pctincomeage25to44.csv', 'income2544')
    income4564 = csvToDict('test_data/pctincomeage45to64.csv', 'income4564')
    income65a = csvToDict('test_data/pctincomeage65a.csv', 'income65a')
    # different method - don't need to cumsum (!)
    tenure = csvToDict_noCumSum('test_data/ownprobability.csv', 'tenure')
    rent1534 = csvToDict('test_data/household1534rent.csv', 'rent1534')
    own1534 = csvToDict('test_data/household1534own.csv', 'own1534')
    rent3564 = csvToDict('test_data/household3564rent.csv', 'rent3564')
    own3564 = csvToDict('test_data/household3564own.csv', 'own3564')
    rent65a = csvToDict('test_data/household65arent.csv', 'rent65a')
    own65a = csvToDict('test_data/household65aown.csv', 'own65a')
    # different method - don't need to cumsum (!)
    children = csvToDict_noCumSum('test_data/pcthouseholdu18.csv', 'children')
    climate = csvToDict_noCumSum('test_data/countychangetest.csv', 'climate')
    bigDict = collections.defaultdict(dict)
    for d in [ageData, u25income, income2544, income4564, income65a, tenure, rent1534, own1534, rent3564, own3564, rent65a, own65a, children, climate]:
        for k, v in d.items():
            bigDict[k].update(v)
    return bigDict