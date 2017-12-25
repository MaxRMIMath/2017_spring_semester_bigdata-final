import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from keras import utils
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

print("load data...")
data="./UCI_Credit_Card.csv"
df=pd.read_csv(data)
df=df.rename(index=str, columns={"PAY_0": "PAY_1"})
print("done")
print()

'''
print("df.describe()...")
print(df.describe())
print()


print("df2.describe()...")
'''
df2=df.astype('int64')
'''
print(df2.describe())
print()
'''
'''
print("compare df df2...")
check=True
for i in df2.columns:
    if df.describe()[i]["std"]!=df2.describe()[i]["std"]:
        check&=False
    else:
        check&=True
print(check)
print()
'''

print("check df2 NaN...")
print(pd.isna(df2).any().any())
print()


print("select PAY as df3, delete PAY as df4...")
df3=df2[df2.columns[6:12]]
df4=df2.drop(df2.columns[6:12], axis=1)
print("done")
print()


print("add PAY column as df4...")
for i in range(1,7):
    df4["PAY_"+str(i)+"_n2"]=pd.Series(df3["PAY_"+str(i)]==-2,index=df4.index)
    df4["PAY_"+str(i)+"_n1"]=pd.Series(df3["PAY_"+str(i)]==-1,index=df4.index)
    df4["PAY_"+str(i)+"_0"]=pd.Series(df3["PAY_"+str(i)]==0,index=df4.index)
    df4["PAY_"+str(i)+"_p"]=pd.Series(df3["PAY_"+str(i)]>0,index=df4.index)
    df4["PAY_"+str(i)+"_AMT"]=pd.Series(df3["PAY_"+str(i)]*(df3["PAY_"+str(i)]>0).astype("int64"),index=df4.index)
print("done")
print()


print("delete ID...")
df4=df4.drop(df4.columns[0], axis=1)
print("done")
print()


print("one hot encodding 'SEX','EDUCATION','MARRIAGE' as X, and default as Y...")
X=pd.get_dummies(df4,columns=["SEX","EDUCATION","MARRIAGE"])
Y=X["default.payment.next.month"].astype('category')
X=X.drop("default.payment.next.month", axis=1).astype("int")
print("done")
print()

print("describe X Y...")
print(X.describe())
print(Y.describe())
#X=X.as_matrix()
#Y=Y.as_matrix()
print("done")
print()


import warnings
from sklearn.feature_selection import SelectKBest, f_classif

warnings.simplefilter(action='ignore', category=(UserWarning,RuntimeWarning))

selector = SelectKBest(f_classif, 25)
selector.fit(X, Y)

top_indices = np.nan_to_num(selector.scores_).argsort()[-25:][::-1]
print(selector.scores_[top_indices])

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)

from sklearn.pipeline import Pipeline
preprocess = Pipeline([('anova', selector), ('scale', scaler)])
preprocess.fit(X,Y)
X_prep = preprocess.transform(X)
X_prep = pd.DataFrame(X_prep)

print("splitting training and testing...")
X_train,X_test,y_train,y_test=train_test_split(X_prep, Y, test_size=0.1, random_state=10)
print("done")
print()

'''
print("oversampling...")
sm = SMOTE(ratio = 1.0)
X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)
print("done")
print()
'''

#y_train_res=utils.to_categorical(Y_train_res,2)


def modelfit(alg, X_train,y_train, X_test, y_test, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train.values, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train, y_train,eval_metric='auc')
        
    #Predict training set:
    X_pred = alg.predict(X_train)
    X_predprob = alg.predict_proba(X_train)[:,1]
    
    X_testpred = alg.predict(X_test)
    X_testpredprob = alg.predict_proba(X_test)[:,1]

    #Print model report:
    print( "\nModel Report")
    print( " - train_acc : %.4g" % metrics.accuracy_score(y_train, X_pred))
    print( " - train auc score: %f" % metrics.roc_auc_score(y_train, X_predprob))
    print( " - text_acc : %.4g" % metrics.accuracy_score(y_test, X_testpred))
    print( " - text auc score: %f" % metrics.roc_auc_score(y_test, X_testpredprob))

'''
predictors = X_train.columns.values.tolist()
xgb1 = XGBClassifier(
 learning_rate =0.2,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, X_train, y_train, X_test, y_test, predictors)

param_test1 = {
 'max_depth':list(range(3,10,2)),
 'min_child_weight':list(range(1,6,2))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch1.fit(X_train,y_train)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
#3,1
'''
'''
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[0,1,2]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch2.fit(X_train,y_train)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
#2,2
'''
'''
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
        }
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=2,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch3.fit(X_train,y_train)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
#0.3
'''
'''
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
        }
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=2,
 min_child_weight=2, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch4.fit(X_train,y_train)
print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)
#0.8,0.8
'''
'''
param_test5 = {
    'subsample':[i/100.0 for i in range(70,95,5)],
    'colsample_bytree':[i/100.0 for i in range(70,95,5)]
        }
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=2,
 min_child_weight=2, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch5.fit(X_train,y_train)
print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
#0.8,0.8
'''
'''
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
        }
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=177, max_depth=2,
 min_child_weight=2, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch6.fit(X_train,y_train)
print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)
#1e-05
'''
'''
param_test7 = {
    'reg_alpha':[5e-07,1e-06,5e-06,1e-05,2e-05,5e-05,1e-04,2e-04,5e-04,1e-03]
        }
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=177, max_depth=2,
 min_child_weight=2, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch7.fit(X_train,y_train)
print(gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_)
#0.0001
'''

predictors = X_train.columns.values.tolist()
xgb2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=2,
 min_child_weight=2,
 gamma=0.3,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 reg_alpha=0.0001,
 seed=27)
modelfit(xgb2, X_train, y_train, X_test, y_test, predictors)

'''
print("print plot...")
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)),loss,label='loss')
plt.plot(range(len(val_loss)),val_loss,label='val_loss')
plt.title('loss')
plt.legend(loc='upper right')
plt.subplot(122)
plt.plot(range(len(acc)),acc,label='acc')
plt.plot(range(len(val_acc)),val_acc,label='val_acc')
plt.title('acc')
plt.legend(loc='lower right')
plt.savefig('origin_network.png',dpi=300,format='png')
plt.close()
print("done")
print()

'''
