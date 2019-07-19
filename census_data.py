import pandas as pd
import censusdata

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

shells = createShells('B11005', 1, 19)
mult = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '41'), ('county', '051')]), shells)
la = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '06'), ('county', '037')]), shells)
bos = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '25'), ('county', '025')]), shells)
dc = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', '11'), ('county', '001')]), shells)
testData = la.append(dc)
testData = testData.append(bos)
testData = testData.append(mult)
testData.to_csv('raw_household65a.csv')