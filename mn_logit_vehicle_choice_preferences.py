'''
 This script is to load and process survey response data to be utilized as an input dataset
 of a multinomial logit model for the CE264 course project in 2017 Spring at UC Berkeley.

 April 7, 2017
 '''

import numpy as np
import pandas as pd
import pylogit as pl


isPreferModel = True # SET FALSE FOR LEAST PREFER CHOICE MODEL
rawFilePath = '/Users/mygreencar/Google Drive/CE264/Project/Data2.0/data.csv' # DESIGNATE YOUR FILE PATH

#========= Attribute data ==========#
# List of attributes dictionary
attrList = [] 
attrList.append({'model':'nissan_leaf','price':30680, 'range':107, 'mpg':112,'annual_cost':600,'parking_avail':0.65,'charging_hrs':6,'type':'EV'})
attrList.append({'model':'toyota_prius','price':26845, 'range':588, 'mpg':52,'annual_cost':650,'parking_avail':0.41,'charging_hrs':0,'type':'PHEV'})
attrList.append({'model':'honda_civic','price':18740, 'range':466, 'mpg':36,'annual_cost':950,'parking_avail':0.32,'charging_hrs':0,'type':'CONV'})
attrList.append({'model':'tesla_s','price':59900, 'range':265, 'mpg':85.6,'annual_cost':750,'parking_avail':0.42,'charging_hrs':12,'type':'EV'})
attrList.append({'model':'jetta_hybrid','price':31670, 'range':524, 'mpg':37.9,'annual_cost':950,'parking_avail':0.67,'charging_hrs':0,'type':'PHEV'})
attrList.append({'model':'benz_c350','price':40865, 'range':348, 'mpg':26.4,'annual_cost':2100,'parking_avail':0.15,'charging_hrs':0,'type':'CONV'})
attrList.append({'model':'bmw_i3','price':42400, 'range':81, 'mpg':136.1,'annual_cost':550,'parking_avail':0.33,'charging_hrs':4,'type':'EV'})
attrList.append({'model':'ford_energi','price':33120, 'range':610, 'mpg':72.1,'annual_cost':750,'parking_avail':0.36,'charging_hrs':0,'type':'PHEV'})
attrList.append({'model':'audi_a4','price':37300, 'range':428, 'mpg':19.5,'annual_cost':1500,'parking_avail':0.45,'charging_hrs':0,'type':'CONV'})
attrList.append({'model':'chevorolet_bolt','price':36620, 'range':238, 'mpg':106,'annual_cost':850,'parking_avail':0.34,'charging_hrs':4.5,'type':'EV'})
attrList.append({'model':'hyundai_sonata','price':34600, 'range':740, 'mpg':40,'annual_cost':850,'parking_avail':0.77,'charging_hrs':0,'type':'PHEV'})
attrList.append({'model':'ford_fusion','price':27830, 'range':438, 'mpg':21,'annual_cost':1400,'parking_avail':0.10,'charging_hrs':0,'type':'CONV'})
attrList.append({'model':'kia_soul','price':32250, 'range':93, 'mpg':105,'annual_cost':600,'parking_avail':0.85,'charging_hrs':4,'type':'EV'})
attrList.append({'model':'cadillac_elr','price':75000, 'range':340, 'mpg':58.7,'annual_cost':950,'parking_avail':0.32,'charging_hrs':0,'type':'PHEV'})
attrList.append({'model':'toyota_camry','price':27810, 'range':568, 'mpg':26.8,'annual_cost':1050,'parking_avail':0.45,'charging_hrs':0,'type':'CONV'})

# Build attributes df
attrDf = pd.DataFrame(columns=attrList[0].keys())
for i in range(len(attrList)):
	attrDf = attrDf.append(pd.DataFrame([attrList[i].values()],columns=attrList[i].keys()),ignore_index=True)

# Models in each decision situation -- NOTE: you add more decision situations here...
modesInDS = {
	1:['nissan_leaf','toyota_prius','honda_civic'],
	2:['tesla_s','jetta_hybrid','benz_c350'],
	3:['bmw_i3','ford_energi','audi_a4'],
	4:['chevorolet_bolt','hyundai_sonata','ford_fusion'],
	5:['kia_soul','cadillac_elr','toyota_camry']
}

#========= Process response data ==========#
# Load raw data
dataRaw = pd.read_csv(rawFilePath)

# Extract questions
data = dataRaw[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12',
              'Q13', 'Q14', 'Q15', 'CS1_1', 'CS1_2', 'CS2_1','CS2_2', 'CS3_1', 'CS3_2', 
              'CS4_1', 'CS4_2', 'CS5_1', 'CS5_2']]

# Remove the first two rows
data = data[2:]

# Filter out data that contains nan data
data = data.dropna(how = 'any',subset = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5','CS1_1', 'CS1_2', 'CS2_1','CS2_2', 'CS3_1', 'CS3_2', 
              'CS4_1', 'CS4_2', 'CS5_1', 'CS5_2'])

# Reset indices
data.reset_index(inplace=True)

# Rename columns
data = data.rename(columns = {'Q1':'Age', 'Q2':'Gender', 'Q3':'Education', 'Q4':'Income', 'Q5':'State', 'Q6':'Parking',
                               'Q6_4_TEXT':'Parking_other', 'Q7':'Daily_range', 'Q8':'Roundtrip', 'Q10':'Current_car',
                               'Q11':'How_many_car', 'Q12':'Mostly_drive_car', 'Q13':'Environment', 'Q14':'Interest_EV',
                               'Q15':'Concern_EV'})

# Add person column and drop index column
data['Person'] = pd.Series(range(0,len(data)))                               
data = data.drop('index',axis=1)

dataPrefer = data.drop(['CS1_2','CS2_2','CS3_2','CS4_2','CS5_2'],axis=1)
dataNoPrefer = data.drop(['CS1_1','CS2_1','CS3_1','CS4_1','CS5_1'],axis=1)

# Convert wide data into long data format for choice sets
choiceSetsPrefer = ['CS1_1','CS2_1','CS3_1','CS4_1','CS5_1']
choiceSetsNoPrefer = ['CS1_2','CS2_2','CS3_2','CS4_2','CS5_2']
# columns = ['Person','Age', 'Gender', 'Education', 'Income', 'State', 'Parking',
#                                'Daily_range','Roundtrip', 'Current_car',
#                                'How_many_car', 'Mostly_drive_car', 'Environment', 'Interest_EV',
#                                'Concern_EV']
# newColumns = ['Person','Age', 'Gender', 'Education', 'Income', 'State', 'Parking',
#                                'Daily_range','Roundtrip', 'Current_car',
#                                'How_many_car', 'Mostly_drive_car', 'Environment', 'Interest_EV',
#                                'Concern_EV','Choice','Decision_set']
columns = ['Person','Age', 'Gender', 'Education', 'Income', 'State', 'Parking',
                               'Daily_range','Roundtrip', 'Current_car',
                               'Environment', 'Interest_EV',
                               'Concern_EV']
newColumns = ['Person','Age', 'Gender', 'Education', 'Income', 'State', 'Parking',
                               'Daily_range','Roundtrip', 'Current_car',
                               'Environment', 'Interest_EV',
                               'Concern_EV','Choice','Decision_set']
if(isPreferModel):
	targetData = dataPrefer
	targetChoiceSets = choiceSetsPrefer
else:
	targetData = dataNoPrefer
	targetChoiceSets = choiceSetsNoPrefer

dataWideTemp = pd.DataFrame(columns = newColumns)
for i in range(len(targetData)):
    for j,choice in enumerate(targetChoiceSets):
        dataWideTemp = dataWideTemp.append(pd.DataFrame([np.append(np.array(targetData.loc[i][columns]),[targetData.loc[i][choice], (j+1)])],columns=newColumns),ignore_index=True)


# Set up data in wide format with attributes for each alternatives
attrColumns = [
	'EV_AV','EV_model','EV_price', 'EV_range','EV_mpg','EV_annual_cost','EV_parking_avail','EV_charging_hrs',
	'PHEV_AV','PHEV_model','PHEV_price', 'PHEV_range','PHEV_mpg','PHEV_annual_cost','PHEV_parking_avail','PHEV_charging_hrs',
	'CONV_AV','CONV_model','CONV_price', 'CONV_range','CONV_mpg','CONV_annual_cost','CONV_parking_avail','CONV_charging_hrs']

# Build Dataframe for decision situations with corresponding attributes
altsWithAttrDf = pd.DataFrame(columns = attrColumns)
for i in range(len(dataWideTemp)): # dataset in wide format
	tempSeries = pd.Series()
	for alts in modesInDS[int(dataWideTemp.loc[i]['Decision_set'])]: # models in each decision situation
		tempSeries = tempSeries.append([
			attrDf[attrDf.model == alts]['price']>=0,attrDf[attrDf.model == alts]['model'],
			attrDf[attrDf.model == alts]['price'],attrDf[attrDf.model == alts]['range'],
			attrDf[attrDf.model == alts]['mpg'],attrDf[attrDf.model == alts]['annual_cost'],
			attrDf[attrDf.model == alts]['parking_avail'],attrDf[attrDf.model == alts]['charging_hrs']],ignore_index=True)
	altsWithAttrDf = altsWithAttrDf.append(pd.DataFrame([tempSeries.tolist()],columns = attrColumns),ignore_index=True)
		
# Build complete data in the wide format by merging alternatives & attributes dataframe to the wide format data frame
dataWide = pd.concat([dataWideTemp, altsWithAttrDf], axis=1)

###  Converting wide data to long data
attrColumnsLong = ['model','price','range','mpg','annual_cost','parking_avail','charging_hrs']

# Index variables
ind_variables = dataWide.columns.tolist()[:13]

alt_varying_variables = {u'model': dict([(1, 'EV_model'),
                                               (2, 'PHEV_model'),
                                               (3, 'CONV_model')]),
						u'price': dict([(1, 'EV_price'),
                                               (2, 'PHEV_price'),
                                               (3, 'CONV_price')]),
						u'range': dict([(1, 'EV_range'),
                                               (2, 'PHEV_range'),
                                               (3, 'CONV_range')]),
                        u'mpg': dict([(1, 'EV_mpg'),
                                               (2, 'PHEV_mpg'),
                                               (3, 'CONV_mpg')]),
                        u'annual_cost': dict([(1, 'EV_annual_cost'),
                                               (2, 'PHEV_annual_cost'),
                                               (3, 'CONV_annual_cost')]),
                        u'parking_avail': dict([(1, 'EV_parking_avail'),
                                               (2, 'PHEV_parking_avail'),
                                               (3, 'CONV_parking_avail')]),
                        u'charging_hrs': dict([(1, 'EV_charging_hrs'),
                                               (2, 'PHEV_charging_hrs'),
                                               (3, 'CONV_charging_hrs')])}
availability_variables = {1: 'EV_AV',
                          2: 'PHEV_AV', 
                          3: 'CONV_AV'}

custom_alt_id = "mode_id"
obs_id_column = "custom_id"
dataWide[obs_id_column] = np.arange(dataWide.shape[0],
                                            dtype=int) + 1

choice_column = "Choice"
dataWide[choice_column] = dataWide[choice_column].astype('int')

dataLong = pl.convert_wide_to_long(dataWide, 
                                           ind_variables, 
                                           alt_varying_variables, 
                                           availability_variables, 
                                           obs_id_column, 
                                           choice_column,
                                           new_alt_id_name=custom_alt_id)

# TODO: ADD YOUR MODEL ESTIMATION SCRIPT HERE
