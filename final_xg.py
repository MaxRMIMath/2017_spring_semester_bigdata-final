import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score


data="./UCI_Credit_Card.csv"
df=pd.read_csv(data)
df=df.rename(index=str, columns={"PAY_0": "PAY_1"})

df2=df.astype('int64')

df3=df2[df2.columns[6:12]]
df4=df2.drop(df2.columns[6:12], axis=1)

for i in range(1,7):
    df4["PAY_"+str(i)+"_n2"]=pd.Series(df3["PAY_"+str(i)]==-2,index=df4.index)
    df4["PAY_"+str(i)+"_n1"]=pd.Series(df3["PAY_"+str(i)]==-1,index=df4.index)
    df4["PAY_"+str(i)+"_0"]=pd.Series(df3["PAY_"+str(i)]==0,index=df4.index)
    df4["PAY_"+str(i)+"_p"]=pd.Series(df3["PAY_"+str(i)]>0,index=df4.index)
    df4["PAY_"+str(i)+"_AMT"]=pd.Series(df3["PAY_"+str(i)]*(df3["PAY_"+str(i)]>0).astype("int64"),index=df4.index)

df4=df4.drop(df4.columns[0], axis=1)

X=pd.get_dummies(df4,columns=["SEX","EDUCATION","MARRIAGE"])
Y=X["default.payment.next.month"].astype('category')
X=X.drop("default.payment.next.month", axis=1).astype("int")

X=X.as_matrix()
Y=Y.as_matrix()

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.1, random_state=10)
sm = SMOTE(ratio={0:len(y_train)-sum(y_train),1:int(4*(len(y_train)-sum(y_train)))})
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train, y_train=shuffle(X_train, y_train)



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
#3,3
'''
'''
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[2,3,4]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch2.fit(X_train,y_train)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
#2,4
'''
'''
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
        }
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=2,
 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch3.fit(X_train,y_train)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
0
'''
'''
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
        }
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=2,
 min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch4.fit(X_train,y_train)
print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)
#0.9,0.6
'''
'''
param_test5 = {
    'subsample':[i/100.0 for i in range(80,100,5)],
    'colsample_bytree':[i/100.0 for i in range(50,70,5)]
        }
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=140, max_depth=2,
 min_child_weight=4, gamma=0, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch5.fit(X_train,y_train)
print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
#0.9,0.6
'''
'''
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
        }
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=177, max_depth=2,
 min_child_weight=4, gamma=0, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch6.fit(X_train,y_train)
print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)
#1
'''
'''
param_test7 = {
    'reg_alpha':[0.1,0.5,1,5,10]
        }
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=177, max_depth=2,
 min_child_weight=4, gamma=0, subsample=0.9, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5,verbose=10)
gsearch7.fit(X_train,y_train)
print(gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_)
#0.5
'''

print("model evaluate...")
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
for train, val in kfold.split(X_train, y_train):

    xgb2 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=5000,
        max_depth=2,
        min_child_weight=4,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.6,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        reg_alpha=0.5)

    xgb2.fit(X_train[train], y_train[train],eval_set = [(X_train[val], y_train[val])],early_stopping_rounds=5,eval_metric='auc',verbose=False)
    y_pred = xgb2.predict(X_train[val])
    scores = accuracy_score(y_train[val], y_pred)
    print("val acc: %.2f%%" % (scores*100))
    cvscores.append(scores * 100)
    

print(" - train acc: %.2f%% (std: %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
y_pred = xgb2.predict(X_test)
score = accuracy_score(y_test, y_pred)
print( " - text acc: %.2f%%" % (score*100))
y_predprob = xgb2.predict_proba(X_test)[:,1]
print( " - text auc score: %f" % metrics.roc_auc_score(y_test, y_predprob))

print(" - confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))


