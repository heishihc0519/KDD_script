import numpy as np
import sys
import os
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,LinearSVR,NuSVR
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import xgboost as xgb

np.set_printoptions(suppress=True)
##############
filepath=sys.argv[1]
f = open(filepath)
dataset = np.loadtxt(f,delimiter=',',skiprows=1)
target=dataset[:,0]
data_train=dataset[:,1:]
targetall=target
data_trainall=data_train

#print len(target)


##evaluation##################################################################

##select feature
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression,k=3)
result=selector.fit(data_train, target)
print result.scores_
x2=selector.transform(data_train)
print x2

##pearson
from scipy.stats import pearsonr
f.seek(0)
listhead=f.readlines()[0].strip().split(',')[1:]
i=0
for line in listhead:
  i=i+1
  print line,pearsonr(dataset[:,i],target)



###train_test_split
target=dataset[0:-504,0]
data_train=dataset[0:-504,1:]
#target=dataset[504:,0]
#data_train=dataset[504:,1:]

if len(sys.argv)==3:  #testing by outside data
 filepath=sys.argv[2]
 f = open(filepath)
 dataset2 = np.loadtxt(f,delimiter=',',skiprows=1)
 test_target=dataset2[:,0]
 test_data=dataset2[:,1:]
else:                #testing by traindata
 test_data =dataset[-504:,1:]
 test_target = dataset[-504:,0]
 #test_data =dataset[0:504,1:]
 #test_target = dataset[0:504,0]
#print len(data_train),len(test_data)

#from sklearn.cross_validation import train_test_split
#data_train,test_data,target,test_target = train_test_split(data_train,target, test_size=0.1, random_state=0)


###method chosen
dictregr=dict()

#dictregr['mlp']=MLPRegressor()
#dictregr['mlpm1000']=MLPRegressor(max_iter=1000)
dictregr['mlpl200']=MLPRegressor(hidden_layer_sizes=(200, ))
#dictregr['mlpl200m1000']=MLPRegressor(hidden_layer_sizes=(200, ),max_iter=1000)
#regr = DecisionTreeRegressor(max_depth=20) 
#dictregr['svr1e3'] = SVR(kernel='rbf', C=1e3)
#dictregr['svrrbf1e3g01'] = SVR(kernel='rbf', C=1e3,gamma=0.1)  #0.3
dictregr['svrrbf'] = SVR(kernel='rbf')  #0.3
#dictregr['svrlinear'] = SVR(kernel='linear')
#dictregr['svrlinearle3'] = SVR(kernel='linear',C=1e3) #0.5
#dictregr['svrpoly'] = SVR(kernel='poly', C=1e3, degree=2)  #take too many time
#dictregr['decisiontree'] = DecisionTreeRegressor()   #0.24
dictregr['decisiontree2'] = DecisionTreeRegressor(max_depth=2)
#dictregr['decisiontree10'] = DecisionTreeRegressor(max_depth=10)

#dictregr['randomforest'] = RandomForestRegressor()
#dictregr['randomforestm5'] = RandomForestRegressor(max_depth=5)
#dictregr['randomforestn100'] = RandomForestRegressor(n_estimators=100)
dictregr['randomforestm5n100'] = RandomForestRegressor(max_depth=5,n_estimators=100)
#dictregr['gb'] = GradientBoostingRegressor()
dictregr['gbv2'] = GradientBoostingRegressor(max_depth=2, learning_rate=0.01,n_estimators=250)
#dictregr['adaboost'] = AdaBoostRegressor()  #0.24

dictregr['bayesian'] = linear_model.BayesianRidge()
#dictregr['sgd']=linear_model.SGDRegressor()
dictregr['ridge'] = linear_model.Ridge()
dictregr['Lasso'] = linear_model.Lasso()
#dictregr['linearSVR'] = LinearSVR()    #0.3
dictregr['linear'] = linear_model.LinearRegression()
dictregr['elasticnet']=linear_model.ElasticNet()

#dictregr['xgb'] = xgb.XGBRegressor()
dictregr['xgbm2n300l001'] = xgb.XGBRegressor(max_depth=2,n_estimators=300,learning_rate=0.01)
#dictregr['xgbm2w5n400l001'] = xgb.XGBRegressor(max_depth=2,min_child_weight=5,n_estimators=400,learning_rate=0.01)
#dictregr['xgbcv'] = GridSearchCV(dictregr['xgb'],{'max_depth': [2,4,6],'min_child_weight':[4,5,6],'n_estimators': [200,300,400],'learning_rate': [0.01,01,0.05]}, verbose=1)

for name,regr in dictregr.items():
  #if name=='bayesian':
  regr.fit(data_train,target)
  #if name=='xgbcv':
  #  print regr.best_estimator_.get_params()
  #if name=='randomforestm5n100':
  #  print regr.feature_importances_
  result=regr.predict(test_data)
  print np.mean(np.abs((test_target - result) / test_target)),regr.score(data_train,target),name

  """
  b0=0
  s0=0
  listb0=list()
  lists0=list()
  for p,t,f in zip(result,test_target,test_data):
     print 'predict:',p, t , abs(p-t), (p-t),f[2],f[3],f[4],f[5],f[6],f[7]
     if (p-t)<0: 
        s0=s0+(p-t)
        lists0.append(p-t)
     else:  
        b0=b0+(p-t)
        listb0.append(p-t)

  print b0,s0
  ndarrayb0= np.array(listb0)
  hist, bins = np.histogram(ndarrayb0, bins=10)
  print hist
  print bins
  ndarrays0= np.array(lists0)
  hist, bins = np.histogram(ndarrays0, bins=10)
  print hist
  print bins
  """


"""
for name,regr in dictregr.items():
  regr.fit(data_train,target)
  #if name=='xgbcv':
  #  print regr.best_estimator_.get_params()
  #if name=='randomforestm5n100':
  #  print regr.feature_importances_
  listmeanvalue=list()
  for test_data_line,test_target_line in zip(test_data,test_target):
   result=regr.predict([test_data_line])
   #for p,t in zip(result,test_target):
   #   print p, t , abs(p-t), (p-t)
   meanvalue=np.mean(np.abs((test_target_line - result) / test_target_line))
   listmeanvalue.append(meanvalue)
   #print np.mean(np.abs((test_target - result) / test_target)),regr.score(data_train,target),name
  print np.average(listmeanvalue),name
"""

#from sklearn.cross_validation import ShuffleSplit,cross_val_score
#cv = ShuffleSplit(len(dataset), n_iter=3, test_size=0.1, random_state=0)
#for name,regr in dictregr.items():
#  test_scores = cross_val_score(regr, data_trainall, targetall, cv=cv, n_jobs=2)  
#  print name,test_scores

##evaluate train data by shufflesplit
#from sklearn.cross_validation import ShuffleSplit,cross_val_score
#cv = ShuffleSplit(len(dataset), n_iter=3, test_size=0.1, random_state=0)
#test_scores = cross_val_score(regr, data_train, target, cv=cv, n_jobs=2)  
#print test_scores

