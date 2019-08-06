import pandas as pd
import censusdata
import collections
import pickle

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

def get_data(fips_dict, table_code, start_shell, end_shell, name):
    shells = createShells(table_code, start_shell, end_shell)
    raw_data = pd.DataFrame()
    for data in fips_dict.values():
        to_append = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', data['state']), ('county', data['county'])]), shells)
        raw_data = raw_data.append(to_append)
    raw_data.to_csv('raw_' + name + '.csv')

def get_single_data(table_code, start_shell, end_shell, name):
    shells = createShells(table_code, start_shell, end_shell)
    data = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '08'), ('county', '041')]), shells)
    data.to_csv('raw_' + name + '.csv')

def make_fips_dict(filepath):
    fips_data = pd.read_csv(filepath, dtype={'name':str, 'state':str, 'county':str})
    fips_dict = fips_data.to_dict('index')
    return fips_dict

def csvToDict(filepath, name):
    ageData = pd.read_csv(filepath)
    cumAgeData = ageData.cumsum(axis=1, skipna=True)
    transAgeData = cumAgeData.transpose()
    ageDataDict = transAgeData.to_dict('list')
    ageData_toMerge = collections.defaultdict(dict)
    for i in range(74):
        ageData_toMerge[i][name] = ageDataDict[i]
    return ageData_toMerge

def csvToDict_noCumSum(filepath, name):
    ageData = pd.read_csv(filepath)
    transAgeData = ageData.transpose()
    ageDataDict = transAgeData.to_dict('list')
    ageData_toMerge = collections.defaultdict(dict)
    for i in range(74):
        ageData_toMerge[i][name] = ageDataDict[i]
    return ageData_toMerge

def mergeDicts():
    ageData = csvToDict('pcthouseholdagedist.csv', 'age')
    u25income = csvToDict('pctincomeu25.csv', 'u25income')
    income2544 = csvToDict('pctincome2544.csv', 'income2544')
    income4564 = csvToDict('pctincome4564.csv', 'income4564')
    income65a = csvToDict('pctincome65a.csv', 'income65a')
    # different method - don't need to cumsum (!)
    tenure = csvToDict_noCumSum('ownprobability.csv', 'tenure')
    """
    rent1534 = csvToDict('test_data/household1534rent.csv', 'rent1534')
    own1534 = csvToDict('test_data/household1534own.csv', 'own1534')
    rent3564 = csvToDict('test_data/household3564rent.csv', 'rent3564')
    own3564 = csvToDict('test_data/household3564own.csv', 'own3564')
    rent65a = csvToDict('test_data/household65arent.csv', 'rent65a')
    own65a = csvToDict('test_data/household65aown.csv', 'own65a')
    # different method - don't need to cumsum (!)
    children = csvToDict_noCumSum('test_data/pcthouseholdu18.csv', 'children')
    """
    climate = csvToDict_noCumSum('countychangemonth.csv', 'climate')
    bigDict = collections.defaultdict(dict)
    for d in [ageData, u25income, income2544, income4564, income65a, tenure, climate]: # rent1534, own1534, rent3564, own3564, rent65a, own65a, children, climate]:
        for k, v in d.items():
            bigDict[k].update(v)
    return bigDict

def make_pickle():
    real_data_dict = mergeDicts()
    print(real_data_dict)
    with open('real_data_dict.pickle', 'wb') as f:
        pickle.dump(real_data_dict, f, pickle.DEFAULT_PROTOCOL)

def get_all_data():
    fips_dict = make_fips_dict('real_data/county_fips.csv')
    get_data(fips_dict, 'B25118', 1, 25, 'tenure')

def get_cumulative_population_list():
    populationData = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    cumPop = populationData.cumsum(axis=0, skipna=True)
    cumDict = cumPop.to_dict('list')
    return cumDict['total_pop_1000']

def get_population_list():
    populationData = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    popDict = populationData.to_dict('list')
    return popDict['total_pop_1000']

# get_single_data('B19037', 36, 69, 'elpasoco4565a')
make_pickle()