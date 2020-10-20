# importer packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle

#lire la dataset

df= pd.read_csv('C:/Users/jesuis/Desktop/train_u6lujuX_CVtuZ9i.csv')
print(df)

                        
print(df.shape)
pd.set_option('display.max_columns',13)
print(df.head())
# voir les valeur manquantes
df.info()

print(df.isnull().sum().sort_values(ascending=False))
#voir s`il ya des valeur pas normal
print(df.describe(include='O'))
# rensegnier les valeur manquantes//deviser mon dataset en 2
cat_data=[]
num_data=[]

print(df.dtypes)

for i, c in enumerate(df.dtypes):
    
      if  c == object:
          
          cat_data.append(df.iloc[:,i])
          
      else: 
          
          num_data.append(df.iloc[:,i])
          
num_data=pd.DataFrame(num_data).transpose()
cat_data=pd.DataFrame(cat_data).transpose()
print(num_data)
print(cat_data)

#pour la base categorique on va remplacer les valeurs manquates par les valeur qui se repetentplus
print(cat_data.isnull().sum().sort_values(ascending=False))
print(num_data.isnull().sum().sort_values(ascending=False))
cat_data=cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
print(cat_data.isnull().sum().any())
#cat_data['Education'].value_counts()
# pour les valeur numerique manquante on va les remplacer par les precedenes
num_data.fillna(method='bfill',inplace=True)
print(num_data.isnull().sum().any())
print(num_data)
print(cat_data)

target_values={'Y':1,'N':0}
target=cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target=target.map(target_values)
print(target)
#replacer les valeur categorique par 0,1,2,....
le= LabelEncoder()
for i in cat_data:
    
    cat_data[i]=le.fit_transform(cat_data[i])

print(cat_data)

#suprimer loan_ID
cat_data.drop('Loan_ID', axis=1, inplace=True)
print(cat_data)
# concatiner cat_data et num_data
x= pd.concat([cat_data,num_data],axis=1)
y= target
print(x)
print(y)












