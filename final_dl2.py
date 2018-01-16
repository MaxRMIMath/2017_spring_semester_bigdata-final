import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import keras
from keras import utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,BatchNormalization,Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import regularizers
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


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


X_train,X_test,Y_train,Y_test=train_test_split(X_prep, Y, test_size=0.1, random_state=10)
sm = SMOTE(ratio={0:len(Y_train)-sum(Y_train),1:int(1.8*(len(Y_train)-sum(Y_train)))})
X_train, Y_train = sm.fit_sample(X_train, Y_train)
X_train, Y_train=shuffle(X_train, Y_train)




print("model evaluate...")
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
for train, val in kfold.split(X_train, Y_train):
    y_train=utils.to_categorical(Y_train[train],2)
    y_val=utils.to_categorical(Y_train[val],2)
    
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2,  activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=['accuracy'])
    
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')


    history=model.fit(X_train[train],y_train,
                validation_data=(X_train[val],y_val),
                batch_size=25,
                epochs=400,
                shuffle=True,
                callbacks=[earlyStopping],
                class_weight ="auto",
                verbose=0
                )
    scores = model.evaluate(X_train[val], y_val, verbose=0)
    print("val acc: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1] * 100)

print(" - train acc: %.2f%% (std: %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
y_test=Y_test
y_predprob=model.predict_proba(X_test, verbose=0)
y_pred = [round(i[1]) for i in y_predprob]
y_predprob = [i[1] for i in y_predprob]
score = accuracy_score(y_test, y_pred)
print( " - text acc: %.2f%%" % (score*100))
print( " - text auc score: %f" % metrics.roc_auc_score(y_test, y_predprob))

print(" - confusion matrix: ")
print(metrics.confusion_matrix(y_test, y_pred))

