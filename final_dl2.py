import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
import keras
from keras import utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,BatchNormalization,Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import regularizers
from imblearn.over_sampling import SMOTE

def Maxout(x, num_unit=None):
    """
    Maxout as in the paper `Maxout Networks <http://arxiv.org/abs/1302.4389>`_.

    Args:
        x (tf.Tensor): a NHWC or NC tensor. Channel has to be known.
        num_unit (int): a int. Must be divisible by C.

    Returns:
        tf.Tensor: of shape NHW(C/num_unit) named ``output``.
    """
    input_shape = x.get_shape().as_list()
    ndim = len(input_shape)
    assert ndim == 4 or ndim == 2

    data_format = keras.backend.image_data_format()

    if data_format == 'channels_first':
        ch = input_shape[1]
    else:
        ch = input_shape[-1]

    if num_unit == None:
        num_unit = ch / 2
    assert ch is not None and ch % num_unit == 0

    if ndim == 4:
        if data_format == 'channels_first':
            x = keras.backend.permute_dimensions(x, (0, 2, 3, 1))
        x = keras.backend.reshape(x, (-1, input_shape[1], input_shape[2], ch // num_unit, num_unit))
        x = keras.backend.max(x, axis=3)
        if data_format == 'channels_first':
            x = keras.backend.permute_dimensions(x, (0, 3, 1, 2))
    else:
        x = keras.backend.reshape(x, (-1, ch // int(num_unit), int(num_unit)))
        x = keras.backend.max(x, axis=1)

    return x


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
X=X.as_matrix()
Y=Y.as_matrix()
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


print("splitting training and testing...")
X_train,X_test,Y_train,Y_test=train_test_split(X_prep, Y, test_size=0.1, random_state=10)
X_train,X_val, Y_train, Y_val=train_test_split(X_train, Y_train, test_size=0.1, random_state=10)
print("done")
print()
'''
print("oversampling...")
sm = SMOTE(ratio  1.0)
X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)
print("done")
print()
'''
#y_train_res=utils.to_categorical(Y_train_res,2)
y_train=utils.to_categorical(Y_train,2)
y_test=utils.to_categorical(Y_test,2)
y_val=utils.to_categorical(Y_val,2)


import logging

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            self.score = roc_auc_score(self.y_val, y_pred)
            print("  -  val auc score:", self.score)

print("deep model...")
'''
model=Sequential()

model.add(Dense(120,input_dim=57))
model.add(BatchNormalization())
model.add(Lambda(Maxout))
model.add(Dropout(0.1))

model.add(Dense(80))
model.add(BatchNormalization())
model.add(Lambda(Maxout))
model.add(Dropout(0.1))


model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Lambda(Maxout))
#model.add(Dropout(0.3))

model.add(Dense(30))
model.add(BatchNormalization())
model.add(Lambda(Maxout))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(BatchNormalization())
model.add(Activation('softmax'))

'''
input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2,  activation='softmax'))

learning_rate = 0.01
decay_rate = learning_rate / 800
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=['accuracy'])
print("done")
print()

ival = IntervalEvaluation(validation_data=(X_train, y_train), interval=1)
print("learning...")
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
history=model.fit(X_train,y_train,
                validation_data=(X_val,y_val),
                batch_size=25,
                epochs=400,
                shuffle=True,
                callbacks=[ival,earlyStopping],
                class_weight ="auto"
                )


loss=history.history.get('loss')
acc=history.history.get('acc')
val_loss=history.history.get('val_loss')
val_acc=history.history.get('val_acc')
test_score= model.evaluate(X_test, y_test, verbose=0)
print(" - test_loss: ",test_score[0]," - test_acc: ",test_score[1])
y_pred=model.predict_proba(X_test, verbose=0)
score=roc_auc_score(y_test , y_pred)
print(" - test auc score: ", score)
print("done")
print()



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


