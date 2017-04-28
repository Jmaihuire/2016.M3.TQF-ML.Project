# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:25:57 2017

@author: zhoukaiwen
"""
#/// import the libraries ********************************************************************************************************************************************************#
import pyodbc
import pandas as pd
import numpy as np 
from numpy import abs 
from numpy import log
from numpy import sign
from scipy.stats import rankdata
from scipy import stats
import math 


#///  query data *****************************************************************************************************************************************************************#
con = pyodbc.connect("DSN=zkw;UID=windpublic;PWD=windpublic")

path ="select s_info_windcode as id, trade_dt as date, s_dq_avgprice as vwap,s_dq_tradestatus as status,s_dq_adjopen as dayopen, s_dq_adjhigh as dayhigh, s_dq_adjlow as daylow, s_dq_adjclose as dayclose,  s_dq_volume as dayvolume, s_dq_pctchange as dayreturn from dbo.ashareeodprices where trade_dt >= 20151131"
result = pd.read_sql(path,con)
result = result.sort(['id','date']).reset_index()
del result['index']

#market cap
path2 ="select s_info_windcode as id, trade_dt as date, s_val_mv as cap from dbo.ashareeodderivativeindicator where trade_dt >= 20110101"
result2 = pd.read_sql(path2,con)
result2 = result2.sort(['id','date']).reset_index()
del result2['index'] 

#industry
path3 ="select s_info_windcode as id, citics_ind_code as ind, entry_dt as date from dbo.ashareindustriesclasscitics "
result3 = pd.read_sql(path3,con)
result3 = result3.sort(['id','date']).reset_index()
del result3['index']

result = pd.merge(result,result2,on=['id','date'],how='inner')
del result2

# merge all
test = result.loc[:,['id','date']]
test2 = pd.merge(result3,test,how='outer').sort(['id','date']).fillna(method='ffill')
result = pd.merge(result,test2,on=['id','date'],how='inner')

# delete useless data
del test 
del test2
del result3


# /// generate factors (10,11,13,16,18,20,21) ************************************************************************************************************************************#
"""
the factors construction methods is unpublic, but I can provide one factor as an example:
factor example: 
rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1))))) 

"""
#**********************************************************************************************************************************************************************************#

# select the in sample data and out of sample data
result['intdate'] = result['date'].apply(lambda x: int(x))
result = result[(result['intdate']>=20160101)]
del result['intdate']

# select the trading day from the database
path3="select trade_days as date from dbo.asharecalendar"
TradingDay = pd.read_sql(path3,con).drop_duplicates() 

# save the data
import os
os.chdir(r'C:\Users\dell\Desktop')

result.to_excel('adaboost2.xlsx')
TradingDay.to_excel('tradeday.xlsx')


#///adaboost method ***************************************************************************************************************************************************************#
"""
Because the data is accessed via SQL database, I only provide some sample data in the file and named it "TestDataSample.xlsx"

Everyone who wants to check or replicate the algorithm should firstly read the sampledata as the following:

"""

"""
import pandas as pd

result = pd.read_excel('TestDataSample.xlsx')
TradingDay = pd.read_excel('tradeday.xlsx')

"""


#////// factor standarlization

# select the factor names
colnames = result.dtypes.index
factornames = [ i for i in colnames if i.startswith('alpha')]

# delete the duplicated date
tradedate = result[['date']].sort('date').reset_index(drop = True)
tradedate = tradedate.drop_duplicates().reset_index(drop = True)

backtestlen = tradedate.shape[0] # the backtest date length
factornamelen = len(factornames) # the factor number

# standarlization
grouped_result = result.groupby(by = 'date')

def standarlization(factor):
  sum_factor = factor.sum()
  return factor/sum_factor

for j in np.arange(factornamelen):
   factor_j = factornames[j]
   result[factor_j] = grouped_result[factor_j].apply(standarlization)
    
# set the dayfrequence and calculate the stock return and market neutral return
DayFrequency = 3
start = 20160101
stop = 20170331

TradingDay2 = list(TradingDay['date'])
TradingDay3 = [pd.Period('%s'%i,freq='D') for i in TradingDay2]

def NearestTradingDay_forward(day): 
   day = pd.Period(day)
   for i in np.arange(len(TradingDay)):
       day = day + i
       if day in TradingDay3:
          return day

def NearestTradingDay_backward(day):
   day = pd.Period(day)
   for i in np.arange(len(TradingDay)):
       day = day - i
       if day in TradingDay3:
          return day
        
DayStart = NearestTradingDay_forward(start)
DayStop = NearestTradingDay_backward(stop)

TestLength = DayStop - DayStart
StartIndex = TradingDay3.index(DayStart)
ChangeDay = []  # change position date 
for i in np.arange(TestLength):
   if StartIndex+i*DayFrequency+1 <= TradingDay3.__len__():
      if TradingDay3[StartIndex+i*DayFrequency] <= DayStop:
         ch_day = TradingDay3[StartIndex+i*DayFrequency]
         ChangeDay.append(ch_day)
         
ChangeDay = list(set(ChangeDay)) # delete the duplicated dates
ChangeLength = ChangeDay.__len__() # change position times
ChangeDay2 = sorted([str(i)[0:4]+str(i)[5:7]+str(i)[8:10] for i in ChangeDay ]) # convert to the normal form date 

test = pd.DataFrame(ChangeDay2,columns=['date']) # generate the change-position date according to the day frequence
test['rankdate']=test['date'].rank(method='dense') 

TradingDay_df = pd.DataFrame(TradingDay2,columns=['date']) 
TradingDay_df['date2'] = TradingDay_df['date'].apply(lambda x: int(x))
TradingDay_df2 = TradingDay_df[(TradingDay_df.date2>=start) & (TradingDay_df.date2<=stop)] # generate all trading date between start and date 
del TradingDay_df2['date2']

datecombine = pd.merge(TradingDay_df2,test,how='left') # fill the all trading date with rank 
datecombine.fillna(method='ffill',inplace=True)

date1 = result[['id','date']]
temp_date = date1['date'].apply(lambda x: int(x))
cond2 = (pd.DataFrame(temp_date.values) <= stop) & (pd.DataFrame(temp_date.values) >= start)
date1 = date1[cond2.values] 
test_result = pd.merge(date1,datecombine,how='left')
test_result = pd.merge(test_result,result,how='left') 
   
test_result3 = test_result[['id','date','dayreturn','rankdate','ind'] + factornames]  # test_result3 is the final data that we use
test_result3['dayreturn'] = test_result3['dayreturn']/100+1
test_temp = test_result3.groupby(['date','ind'])
test_temp2 = pd.DataFrame(test_temp['dayreturn'].mean()-1) 
test_temp2 = test_temp2.rename(columns={'dayreturn':'indreturn'})
test_temp2['ind'] = test_temp2.index.get_level_values('ind') 
test_temp2['date'] = test_temp2.index.get_level_values('date') 
test_result3 = pd.merge(test_result3,test_temp2,how='left')
test_result3['dayreturn_ind'] = test_result3['dayreturn'] - test_result3['indreturn'] 

grouped_data = test_result3.groupby([test_result3['id'],test_result3['rankdate']]) 
mon_ret=grouped_data['dayreturn'].prod()-1  # generate the interval pure return 
mon_ret=mon_ret.reset_index()
mon_ret.loc[mon_ret['dayreturn'].isnull(),'dayreturn']=0
mon_ret.rename(columns={'dayreturn':'periodreturn'},inplace = True)

mon_ret2=grouped_data['dayreturn_ind'].prod()-1  # generate the interval return(delete the industry average)
mon_ret2=mon_ret2.reset_index()
mon_ret2.loc[mon_ret2['dayreturn_ind'].isnull(),'dayreturn_ind']=0
mon_ret2.rename(columns={'dayreturn_ind':'periodreturn_ind'},inplace = True)

last_day = pd.DataFrame(test_result3.groupby([test_result3['rankdate']]) ['date'].max(),columns=['date']) # generate the last day of each interval
last_day = last_day.sort('date').reset_index()

finalresult = pd.merge(last_day,test_result3,how ='inner') # generate the final backtest data
finalresult = pd.merge(finalresult,mon_ret,how = 'inner')
finalresult = pd.merge(finalresult,mon_ret2,how = 'inner') 
finalresult = finalresult.sort(['id','rankdate']).reset_index(drop = True)

groupedfinalresult = finalresult.groupby(['id'])  # generate the shift data
def shiftdata(df):
  df['shift_periodreturn'] = df['periodreturn'].shift(periods=-1)
  df['shift_periodreturn_ind'] =df['periodreturn_ind'].shift(periods=-1)
  return df
finalresult = groupedfinalresult.apply(shiftdata)
finalresult = finalresult.sort(['rankdate','id']).reset_index(drop = True)

# adaboosting kernal methods (combine the old factors into one new factor)
 # method 1
  # method description : just bumping method , equal weighted all factors to generate a new factor
finalresult['adaboost1'] = 0
for j in np.arange(factornamelen):
  finalresult['adaboost1'] = finalresult['adaboost1'] + finalresult[factornames[j]]

 # method 2
  # method description : use the normal adaboosting method (factors order is random , so there is a penalty on the coefficient)
backtestdate = finalresult['date'].drop_duplicates().reset_index(drop = True)
# backtestdate = backtestdate.drop([backtestdate.index.max()]) # delete the last date due to no period data 
backtestdatelen = len(backtestdate)

dateparts = locals() # generate the local var sets 
for j in np.arange(backtestdatelen): # generate the different var name date'j'
  dateparts['date%s'% j] = finalresult[finalresult['date']==backtestdate[j]]

def renewweight(df):
  if df['factorgroup'] != df['returngroup']:
    df['weight'] = df['weight'] * math.exp(errorrate)
  else:
    df['weight'] = df['weight'] * math.exp(-errorrate)
  return df

for j in np.arange(backtestdatelen-1):
  locals()['date'+str(j+1)]['adaboost2'] = 0 
  stocknumber = locals()['date'+str(j)].shape[0]
  locals()['date'+str(j)]['weight'] = 1 / stocknumber   # initial the weight
  for i in np.arange(factornamelen):
    factor = factornames[i]
    locals()['date'+str(j)] = locals()['date'+str(j)][locals()['date'+str(j)][factor].notnull()] # delete the null factor
    locals()['date'+str(j)]['factorrank'] = locals()['date'+str(j)][factor].rank(method='dense')
    if i<6 : # these factors are normal factors (have lots of values)
      factorgroup = locals()['date'+str(j)][['factorrank']].drop_duplicates('factorrank').rank(method='dense').sort('factorrank').shape[0]
      locals()['date'+str(j)]['factorgroup'] = locals()['date'+str(j)]['factorrank'].map(lambda x: math.ceil(x/factorgroup*5)) # group 5 according to the factor
      locals()['date'+str(j)]['shift_periodreturn'] = locals()['date'+str(j)][['shift_periodreturn']].fillna(value = 0) # fill the periodreturn nan = 0
      locals()['date'+str(j)]['returnrank'] = locals()['date'+str(j)]['shift_periodreturn'].rank(method='dense')
      locals()['date'+str(j)]['returngroup'] = locals()['date'+str(j)]['returnrank'].map(lambda x: math.ceil(x/factorgroup*5)) # group 5 according to the next period return
    else : # this factor('alpha021') is a 0-1 factor (only have two values, so can only be divided into 2 groups)
      factorgroup = locals()['date'+str(j)][['factorrank']].drop_duplicates('factorrank').rank(method='dense').sort('factorrank').shape[0]
      locals()['date'+str(j)]['factorgroup'] = locals()['date'+str(j)]['factorrank'].map(lambda x: math.ceil(x/factorgroup*2)) # group 2 according to the factor
      locals()['date'+str(j)]['shift_periodreturn'] = locals()['date'+str(j)][['shift_periodreturn']].fillna(value = 0) # fill the periodreturn nan = 0
      locals()['date'+str(j)]['returnrank'] = locals()['date'+str(j)]['shift_periodreturn'].rank(method='dense')
      locals()['date'+str(j)]['returngroup'] = locals()['date'+str(j)]['returnrank'].map(lambda x: math.ceil(x/factorgroup*2)) # group 2 according to the next period return
    errorrate = locals()['date'+str(j)].where(locals()['date'+str(j)]['factorgroup'] != locals()['date'+str(j)]['returngroup']).dropna().shape[0]/stocknumber # to renew the weight
    weighterrorrate = locals()['date'+str(j)].where(locals()['date'+str(j)]['factorgroup'] != locals()['date'+str(j)]['returngroup']).dropna()['weight'].sum() # to renew the adaboost factor
    
    locals()['date'+str(j)] = locals()['date'+str(j)].apply(renewweight,axis =1) # renew the weight
    locals()['date'+str(j)]['weight'] = locals()['date'+str(j)]['weight'] / locals()['date'+str(j)]['weight'].sum() # weight uniform

    locals()['date'+str(j+1)]['adaboost2'] =locals()['date'+str(j+1)]['adaboost2'] + 1/weighterrorrate * locals()['date'+str(j+1)][factor] # renew the adaboost
   
  locals()['date'+str(j)] = locals()['date'+str(j)].drop(['weight','factorrank','factorgroup','returnrank','returngroup'],axis =1)

finalresult2 = locals()['date'+str(1)] # append all data to become finalresult2 and delete all the temporary datej
for j in np.arange(backtestdatelen-2):
  finalresult2 = finalresult2.append(locals()['date'+str(j+2)])
  del locals()['date'+str(j+2)]
del locals()['date'+str(0)]
del locals()['date'+str(1)]


 # method 3
  # method description : use the factor according to the minimum error rate ( nonrandom order , so there is no penalty coefficient  )
backtestdate = finalresult2['date'].drop_duplicates().reset_index(drop = True)
#backtestdate = backtestdate.drop([backtestdate.index.max()]) # delete the last date due to no period data 
backtestdatelen = len(backtestdate)

dateparts = locals() # generate the local var sets 
for j in np.arange(backtestdatelen): # generate the different var name date'j'
  dateparts['date%s'% j] = finalresult2[finalresult2['date']==backtestdate[j]]
  
def renewweight(df):
  if df['factorgroup'] != df['returngroup']:
    df['weight'] = df['weight'] * math.exp(errorrate)
  else:
    df['weight'] = df['weight'] * math.exp(-errorrate)
  return df


for j in np.arange(backtestdatelen-1): # j represents backtestdate index

  locals()['date'+str(j+1)]['adaboost3'] = 0 
  stocknumber = locals()['date'+str(j)].shape[0]
  locals()['date'+str(j)]['weight'] = 1 / stocknumber   # initial the weight
  G1G5weightsum = [] # group1 + group5 weights
  index = [] # record the factor has been used
  k = 0
  
  # generate G1G5weightsum and index and update adaboost3
  while k <= factornamelen-1:  # k represents G1G5weightsum index
    if k < factornamelen-1:
      
      i = 0  # i represents factor index
      tempweightsum = [] # record all the weightsum
      tempindex = [] # record all the index
      while i < factornamelen-1:
        if i in index:
          i = i+1 # i has been recorded in index 
        else :
          factor = factornames[i]
          locals()['date'+str(j)] = locals()['date'+str(j)][locals()['date'+str(j)][factor].notnull()] # delete the null factor
          locals()['date'+str(j)]['factorrank'] = locals()['date'+str(j)][factor].rank(method='dense')
          factorgroup = locals()['date'+str(j)][['factorrank']].drop_duplicates('factorrank').rank(method='dense').sort('factorrank').shape[0]
          locals()['date'+str(j)]['factorgroup'] = locals()['date'+str(j)]['factorrank'].map(lambda x: math.ceil(x/factorgroup*5)) # group 5 according to the factor
          G1G5weightsum_i = locals()['date'+str(j)].where( (locals()['date'+str(j)]['factorgroup']==1) | (locals()['date'+str(j)]['factorgroup']==5)).dropna()[['weight']].sum()
          tempweightsum.append(G1G5weightsum_i)
          tempindex.append(i)
          i = i+1
      # find the best factor 
      tempweightsumindex = np.where(pd.DataFrame(tempweightsum) == pd.DataFrame(tempweightsum).min())[0][0]
      factorindex = tempindex[tempweightsumindex]
      G1G5weightsum.append(tempweightsum[tempweightsumindex])
      index.append(factorindex)
      # update the weight
      factor = factornames[factorindex]
      locals()['date'+str(j)] = locals()['date'+str(j)][locals()['date'+str(j)][factor].notnull()] # delete the null factor
      locals()['date'+str(j)]['factorrank'] = locals()['date'+str(j)][factor].rank(method='dense')
      factorgroup = locals()['date'+str(j)][['factorrank']].drop_duplicates('factorrank').rank(method='dense').sort('factorrank').shape[0]
      locals()['date'+str(j)]['factorgroup'] = locals()['date'+str(j)]['factorrank'].map(lambda x: math.ceil(x/factorgroup*5)) # group 5 according to the factor
      locals()['date'+str(j)]['shift_periodreturn'] = locals()['date'+str(j)][['shift_periodreturn']].fillna(value = 0) # fill the periodreturn nan = 0
      locals()['date'+str(j)]['returnrank'] = locals()['date'+str(j)]['shift_periodreturn'].rank(method='dense')
      locals()['date'+str(j)]['returngroup'] = locals()['date'+str(j)]['returnrank'].map(lambda x: math.ceil(x/factorgroup*5)) # group 5 according to the next period return
      errorrate = locals()['date'+str(j)].where(locals()['date'+str(j)]['factorgroup'] != locals()['date'+str(j)]['returngroup']).dropna().shape[0]/stocknumber # to renew the weight
      locals()['date'+str(j)] = locals()['date'+str(j)].apply(renewweight,axis =1) # renew the weight
      locals()['date'+str(j)]['weight'] = locals()['date'+str(j)]['weight'] / locals()['date'+str(j)]['weight'].sum() # weight uniform
      # update the adaboost3
      locals()['date'+str(j+1)]['adaboost3'] =locals()['date'+str(j+1)]['adaboost3'] +  locals()['date'+str(j+1)][factor] # renew the adaboost
      
      k = k+1
      
    else:  # k==6 , to deal with the 0-1 variable
      factor = factornames[k]
      locals()['date'+str(j)] = locals()['date'+str(j)][locals()['date'+str(j)][factor].notnull()] # delete the null factor
      locals()['date'+str(j)]['factorrank'] = locals()['date'+str(j)][factor].rank(method='dense')
      factorgroup = locals()['date'+str(j)][['factorrank']].drop_duplicates('factorrank').rank(method='dense').sort('factorrank').shape[0]
      locals()['date'+str(j)]['factorgroup'] = locals()['date'+str(j)]['factorrank'].map(lambda x: math.ceil(x/factorgroup*2)) # group 2 according to the factor
      locals()['date'+str(j)]['shift_periodreturn'] = locals()['date'+str(j)][['shift_periodreturn']].fillna(value = 0) # fill the periodreturn nan = 0
      locals()['date'+str(j)]['returnrank'] = locals()['date'+str(j)]['shift_periodreturn'].rank(method='dense')
      locals()['date'+str(j)]['returngroup'] = locals()['date'+str(j)]['returnrank'].map(lambda x: math.ceil(x/factorgroup*2)) # group 2 according to the next period return
      
      errorrate = locals()['date'+str(j)].where(locals()['date'+str(j)]['factorgroup'] != locals()['date'+str(j)]['returngroup']).dropna().shape[0]/stocknumber # to renew the weight
      locals()['date'+str(j+1)]['adaboost3'] =locals()['date'+str(j+1)]['adaboost3'] + 1/errorrate * locals()['date'+str(j+1)][factor] # renew the adaboost
      
      k = k+1
  
  #print(j)
  locals()['date'+str(j)] = locals()['date'+str(j)].drop(['weight','factorrank','factorgroup','returnrank','returngroup'],axis =1)

finalresult3 = locals()['date'+str(1)] # append all data to become finalresult2 and delete all the temporary datej
for j in np.arange(backtestdatelen-2):
  finalresult3 = finalresult3.append(locals()['date'+str(j+2)])
  del locals()['date'+str(j+2)]
del locals()['date'+str(0)]
del locals()['date'+str(1)]

# /// backtest the adaboost factor and compare them ( use finalresult3 )

# choose the test factor
factor = 'adaboost3'

# generate backtest table
backtest_result = last_day.copy(deep=True)
backtest_result = backtest_result.drop([0,1]).reset_index(drop = True)  # delete the first two rows date due to adaboost data use the t+1 information , only t+2 can start
backtest_result = backtest_result.drop(backtest_result.shape[0]-1).reset_index(drop = True)

# generate IC column
backtest_result['IC'+factor]=0
# generate each group performance
for j in np.arange(5):
   col_name='group'+str(j+1)
   backtest_result[col_name]=0
# generate each group excess market performance
for j in np.arange(5):
   col_name='group_ind'+str(j+1)
   backtest_result[col_name]=0
# generate each group turnover rate
for j in np.arange(5):
   col_name='group'+str(j+1)+'_to'
   backtest_result[col_name]=0

# factor test main body
test_len = len(backtest_result)
for i in np.arange(test_len):

       date_i = backtest_result.loc[i,'rankdate'] # select the last date
       result_i = finalresult3[finalresult3['rankdate']==date_i] # select the date_i data
       # result_i = result_i[result_i['status']=='交易'] # deletet the suspended stock

       # group data
       result_i = result_i[result_i[factor].notnull()] # delete the null factor
       result_i['shift_periodreturn'] = result_i[['shift_periodreturn']].fillna(value = 0) # fill the periodreturn nan = 0
       result_i['shift_periodreturn_ind'] = result_i[['shift_periodreturn_ind']].fillna(value = 0) 
       result_i['periodreturn'] = result_i[['periodreturn']].fillna(value = 0) 
       result_i['periodreturn_ind'] = result_i[['periodreturn_ind']].fillna(value = 0)
       
       if len(result_i)>0: 
         
          result_i['factorrank'] = result_i[factor].rank(ascending= True,method='dense') # factor rank
          stk_num = result_i.index.size 
          factorgroup = result_i[['factorrank']].drop_duplicates('factorrank').rank(method='dense').sort('factorrank').shape[0]
          result_i['factorrank'] = result_i['factorrank'].map( lambda x:math.ceil(x/factorgroup*5) ) # divide into 5 groups
          
          #IC
          if not result_i.empty:
             cov=np.cov(result_i[[factor,'shift_periodreturn']].T)[0,1]
             backtest_result.loc[i,'IC'+factor]=cov/(result_i[factor].std()*(result_i['shift_periodreturn'].std()))

          #portfolio
          ave_ret = result_i.groupby('factorrank')['shift_periodreturn'].mean()
          backtest_result.loc[i,'group1':'group5']=ave_ret[0:5].values
          
          #portfolio的市场中性后的收益
          ave_ret2 = result_i.groupby('factorrank')['shift_periodreturn_ind'].mean()
          backtest_result.loc[i,'group_ind1':'group_ind5']=ave_ret2[0:5].values

          #turnover
          if i>=1:
              unchnged_stk = pd.merge(port_list,result_i,on=['id','factorrank'],how='inner')
              stk_num_lastmon = port_list.groupby('factorrank').count()
              stk_num_unchnged = unchnged_stk.groupby('factorrank').count()
              to=1-stk_num_unchnged['id'].div(stk_num_lastmon.id,axis=0)
              backtest_result.loc[i,'group1_to':'group5_to']=to.values
          port_list = result_i[['id','factorrank']]


# /// export the data to files
backtest_result.to_excel(factor+'result.xlsx')
