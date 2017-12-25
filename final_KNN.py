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
X_train,X_val,y_train,y_val=train_test_split(X_train, y_train, test_size=0.1, random_state=10)
print("done")
print()

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def modelfit(X_train,y_train, X_val,y_val,X_test, y_test):
    cal=0
    pos=0
    '''
    for i in range(1,200,20):
        model = KNeighborsClassifier(n_neighbors=i)

        #Fit the algorithm on the data
        model.fit(X_train, y_train)

        #Predict training set:
        y_pred = model.predict(X_train)
        y_predprob = model.predict_proba(X_train)[:,1]

        y_testpred = model.predict(X_val)
        y_testpredprob = model.predict_proba(X_val)[:,1]
        
        if metrics.accuracy_score(y_val, y_testpred)+1.5*metrics.roc_auc_score(y_val, y_testpredprob)>cal:
            pos = i
            cal = metrics.accuracy_score(y_val, y_testpred)+1.5*metrics.roc_auc_score(y_val, y_testpredprob)

        print( "\n",i," Model Report")
        print( " - train_acc : %.4g" % metrics.accuracy_score(y_train, y_pred))
        print( " - train auc score: %f" % metrics.roc_auc_score(y_train, y_predprob))
        print( " - text_acc : %.4g" % metrics.accuracy_score(y_val, y_testpred))
        print( " - text auc score: %f" % metrics.roc_auc_score(y_val, y_testpredprob))
        print("\n",pos," is the best")
    '''
    model = KNeighborsClassifier(n_neighbors=83)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_predprob = model.predict_proba(X_train)[:,1]

    y_testpred = model.predict(X_test)
    y_testpredprob = model.predict_proba(X_test)[:,1]

    print( "\n83 Model Report")
    print( " - train_acc : %.4g" % metrics.accuracy_score(y_train, y_pred))
    print( " - train auc score: %f" % metrics.roc_auc_score(y_train, y_predprob))
    print( " - text_acc : %.4g" % metrics.accuracy_score(y_test, y_testpred))
    print( " - text auc score: %f" % metrics.roc_auc_score(y_test, y_testpredprob))



modelfit(X_train,y_train, X_val, y_val, X_test,y_test)



