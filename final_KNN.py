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
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


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

selector = SelectKBest(f_classif, 25)
scaler = preprocessing.MinMaxScaler()

preprocess = Pipeline([('anova', selector), ('scale', scaler)])
preprocess.fit(X,Y)
X_prep = preprocess.transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_prep, Y, test_size=0.1, random_state=10)
sm = SMOTE(ratio={0:len(y_train)-sum(y_train),1:int(1.7*(len(y_train)-sum(y_train)))})
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train, y_train=shuffle(X_train, y_train)




print("model evaluate...")
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []

for train, val in kfold.split(X_train, y_train):
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
    model.fit(X_train[train], y_train[train])
    y_pred = model.predict(X_train[val])
    scores = accuracy_score(y_train[val], y_pred)
    print("val acc: %.2f%%" % (scores*100))
    cvscores.append(scores * 100)


print(" - train acc: %.2f%% (std: %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
y_predprob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print( " - text acc: %.2f%%" % (score*100))
print( " - text auc score: %f" % roc_auc_score(y_test, y_predprob))

print(" - confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))


