# importer packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

# commencer par la varieble target
print(target.value_counts())
#toute la base de donne utilise

df1=pd.concat([cat_data,num_data,target],axis=1)
print(df1)

#visualisation (maniere categorique)
plt.figure(figsize= (6,8))
sns.countplot(target)
yes=target.value_counts()[1]/len(target)
no= target.value_counts()[0]/len(target)
print(f'le pourcentage des credit accordes est :{yes}')
print(f'le pourcentage des credit non accordes est :{no}')

#historique de credit 
grid= sns.FacetGrid(df1, col='Loan_Status',size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History')

#sex 
grid= sns.FacetGrid(df1, col='Loan_Status',size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender')

#MMarier ou pas  
grid= sns.FacetGrid(df1, col='Loan_Status',size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Married')

#Education  
grid= sns.FacetGrid(df1, col='Loan_Status',size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Education')

#revenu de demandeur (visualization pour numerique)
plt.scatter(df1['ApplicantIncome'],df1['Loan_Status'])

#revenu de demandeur conjoint (visualization pour numerique)
plt.scatter(df1['CoapplicantIncome'],df1['Loan_Status'])

df1.groupby('Loan_Status').median()


#developpement du model 
# machine learning

#diviser notre base de donne entre base de donne d`entrainement et autre de test

sss= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train ,test in sss.split(x,y):
   x_train, x_test = x.iloc[train], x.iloc[test]   
   y_train, y_test = y.iloc[train], y.iloc[test] 

print('x_train taille: ', x_train.shape)
print('x_test taille: ', x_test.shape)    
print('y_train taille: ', y_train.shape)
print('y_test taille: ', y_test.shape) 

#appliquier notre model /// 3 algorithme machine learniing
#// logistic Regresion. KNN.DecisionTree

models={
       
       'LogisticRegression':LogisticRegression(random_state=42),
       'KNeighborsClassifier':KNeighborsClassifier(),
       'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1, random_state=42)
       
       }
#definir la fonction de precision 
def accu(y_true,y_pred, retu= False):
     acc= accuracy_score(y_true,y_pred)
     if retu:
         return acc
     else:
         print(f'la precision de model est : {acc}')
         
#fonction d`application de model 

def train_test_eval(models,x_train,y_train,x_test,y_test):
 for name, model in models.items():
     
     print(name, ':')
     model.fit(x_train, y_train)
     accu(y_test,model.predict(x_test))
     print('-'*30)
     
train_test_eval(models, x_train, y_train, x_test, y_test)

# une base de donne pour appliquer le model 
x_2= x[['Credit_History','Married', 'CoapplicantIncome']]

sss= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train ,test in sss.split(x_2,y):
   x_2_train, x_2_test = x_2.iloc[train], x_2.iloc[test]   
   y_train, y_test = y.iloc[train], y.iloc[test] 

print('x_2_train taille:', x_2_train.shape)
print('x_2_test taille:', x_2_test.shape)    
print('y_train taille:', y_train.shape)
print('y_test taille:', y_test.shape) 

train_test_eval(models, x_2_train, y_train, x_2_test, y_test)






























