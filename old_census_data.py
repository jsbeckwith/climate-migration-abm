import pandas as pd
import censusdata

pd.set_option('display.expand_frame_repr', False)

def createAgeShells():
    age_shells = []
    for i in range(1 ,50):
        shell = 'B01001_0'
        if i < 10:
            shell += str(0)
        shell += str(i)
        shell += 'E'
        age_shells.append(shell)
    return age_shells

def getAgeData(age_shells, state_fips, county_fips):
    age_data = censusdata.download('acs5', 2013, censusdata.censusgeo([('state', state_fips), ('county', county_fips)]), age_shells)
    return age_data

def getAdults(age_data):
    mtotal = int(age_data.B01001_002E)
    ftotal = int(age_data.B01001_026E)

    for j in range(3, 7):
        shell = 'B01001_00'
        shell += str(j) + 'E'
        mtotal -= int(getattr(age_data, shell))

    for k in range(27, 31):
        shell = 'B01001_0'
        shell += str(k) + 'E'
        ftotal -= int(getattr(age_data, shell))

    age_data['m_total'] = mtotal
    age_data['f_total'] = ftotal
    age_data['total_18+'] = mtotal + ftotal
    
    return age_data

def ageBuckets(age_data, sex):
    if sex == 'm':
        shell_num = 7
    else:
        shell_num = 31
    k = 1
    for i in [4, 4, 4, 6]:
        j = 0
        bucket = 0
        while j < i:
            shell = 'B01001_0'
            if shell_num < 10:
                shell += '0'
            shell += str(shell_num) + 'E'
            bucket += int(getattr(age_data, shell))
            shell_num += 1
            j += 1
        if sex == 'm':
            bucketName = 'mbucket' + str(k)
        else:
            bucketName = 'fbucket' + str(k)
        age_data[bucketName] = bucket
        k += 1
    
    return age_data

def agePercentage(age_data):
    for i in ['m', 'f']:
        for j in range(1, 5):
            accessBucketName = i + 'bucket' + str(j)
            totalBucket = i + '_total'
            age_data[i + 'pct' + str(j)] = getattr(age_data, accessBucketName) / getattr(age_data, totalBucket)
    return age_data

def cleanData(age_data):
    age_data = age_data[['m_total', 'f_total', 'total_18+', 
                        'mbucket1', 'mbucket2', 'mbucket3', 'mbucket4', 
                        'fbucket1', 'fbucket2', 'fbucket3', 'fbucket4', 
                        'mpct1', 'mpct2', 'mpct3', 'mpct4', 
                        'fpct1', 'fpct2', 'fpct3', 'fpct4']]
    return age_data

def cleanAgeData(state_fips, county_fips):
    shells = createAgeShells()
    ageData = getAgeData(shells, state_fips, county_fips)
    ageData = getAdults(ageData)
    ageData = ageBuckets(ageData, 'm')
    ageData = ageBuckets(ageData, 'f')
    ageData = agePercentage(ageData)
    ageData = cleanData(ageData)
    ageDataList = ageData.to_dict('list')
    for key in ageDataList.keys():
        ageDataList[key] = ageDataList[key][0]
    print(ageDataList)
    return ageDataList

"""
mult = cleanAgeData('41', '051')
mult['name'] = 'multnomah'
la = cleanAgeData('06', '037')
la['name'] = 'losangeles'
bos = cleanAgeData('25', '025')
bos['name'] = 'suffolk'
dc = cleanAgeData('11', '001')
dc['name'] = 'dc'

cityAttributes = {0: la, 1: dc, 2: bos, 3: mult}

cityAttributesdf = pd.DataFrame.from_dict(cityAttributes, orient='index')
cityAttributesdf.to_csv("cityAttributes2.csv")
climateData = pd.read_csv('countychangetest.csv')
"""

def mergeClimateDemographic():
    climateData = pd.read_csv('countychangetest.csv')
    demoData = pd.read_csv('cityAttributes2.csv')
    mergeData = pd.merge(demoData, climateData, on='id')
    mergeDataDict = mergeData.to_dict('index')
    return(mergeDataDict)