import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
pd.options.display.max_columns=100
pd.options.display.max_rows=100
import seaborn as sns

plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.rcParams['font.size']=100
plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'

plt.style.use('ggplot')

#print("load data...")
data="./UCI_Credit_Card.csv"
df=pd.read_csv(data)
df=df.rename(index=str, columns={"PAY_0": "PAY_1"})
print("\nany na...")
print(pd.isna(df).any().any(),"\n")
df2=df.astype('int64')
for i in ['SEX','EDUCATION','MARRIAGE',"default.payment.next.month"]:
    df2[i]=df2[i].astype('category')


for i in df2.keys():
    if i=="ID":
        continue
    if i in ['SEX','EDUCATION','MARRIAGE',"default.payment.next.month"]:
        print(df2[i].describe(include='all'),"\n")
        dic={}
        for value in df2[i]:
            if value not in dic:
                dic[int(value)]=1
            else:
                dic[int(value)]+=1
        keys=dic.keys()
        plt.bar(keys,dic.values(),tick_label=list(keys))
    else:
        print(df2[i].describe(),"\n")
        plt.hist(df2[i])
    plt.ylabel('Amount');
    plt.title(i)
    plt.savefig("./image/"+i+".png")
    plt.close()


#print("select PAY as df3, delete PAY as df4...")
df3=df2[df2.columns[6:12]]
df4=df2.drop(df2.columns[6:12], axis=1)
#print("add PAY column as df4...")
for i in range(1,7):
    df4["PAY_"+str(i)+"_n2"]=pd.Series(df3["PAY_"+str(i)]==-2,index=df4.index)
for i in range(1,7):
    df4["PAY_"+str(i)+"_n1"]=pd.Series(df3["PAY_"+str(i)]==-1,index=df4.index)
for i in range(1,7):
    df4["PAY_"+str(i)+"_0"]=pd.Series(df3["PAY_"+str(i)]==0,index=df4.index)
for i in range(1,7):
    df4["PAY_"+str(i)+"_p"]=pd.Series(df3["PAY_"+str(i)]>0,index=df4.index)
for i in range(1,7):
    df4["PAY_"+str(i)+"_AMT"]=pd.Series(df3["PAY_"+str(i)]*(df3["PAY_"+str(i)]>0).astype("int64"),index=df4.index)
#print("delete ID...")
df4=df4.drop(df4.columns[0], axis=1)
#print("one hot encodding 'SEX','EDUCATION','MARRIAGE' as X, and default as Y...")
X=pd.get_dummies(df4,columns=["SEX","EDUCATION","MARRIAGE"])
Y=X["default.payment.next.month"].astype('uint8')
X=X.drop("default.payment.next.month", axis=1)

#print("describe X Y...")
print("\nX describe...")
print(X.describe())
print("\nY describe...")
print(Y.describe())

selector = SelectKBest(f_classif, 25)
select_k_best_classifier=selector.fit_transform(X, Y)
mask = selector.get_support()
feature=X.keys()[mask]
print("\nfeature...")
print(feature)

scaler = preprocessing.MinMaxScaler()
X_prep=scaler.fit_transform(select_k_best_classifier,Y)
print("\n after transform X_prep...")
print(X_prep)


corr=X.join(Y).corr()

plt.rcParams['font.size']=10
sns.heatmap(corr).get_figure().savefig("./image/output.png")
plt.close()


