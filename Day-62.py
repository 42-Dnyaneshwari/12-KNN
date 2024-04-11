# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:45:30 2024

@author: Dnyaneshwari
"""
#KNN

import pandas as pd
import numpy as np 
wbcd = pd.read_csv('wbcd.csv')
#Tere are 569 rows and 32 columns 
wbcd.describe()
#in output columns there is only B for Benien and M for Malignant 
#Let us first convert it as Benien and Malignant 
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B', 'Beniegn', wbcd['diagnosis'])
#In wbcd there is ccolumn named 'diagnosis', where ever threr is 'B' replace it with Benien 
#Similarly for M = Malignant
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M', 'Malignant', wbcd['diagnosis'])
########################################################################33 
#0th Column is patient ID let us drop it 
wbcd = wbcd.iloc[:,1:32]
#########################################################################3
#Normalisation 
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())
#Now let us apply this normalisation function to the dataframe 
wbcd_n = norm_func(wbcd.iloc[:,1:32])
#becaue now 0th column is output or lable it is not considered hence l
####################################################################### 
#Let us now apply X as input and Y as Output  
X = np.array(wbcd_n.iloc[:,:])
#since in wbcd_n, we are already excluding output column, hence all rows and columns 
y = np.array(wbcd['diagnosis'])
##############################################################33 
#Now let us split the data into training and testing 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#Here you are passing X,y instea dataframe handle 
#there could be chances of unbalancing of data 
#let us assume you have 100 data points, out of which 80 NC and 20 cancer 
#These data points must be equally distributed 
#ther is satisfied sampling concept is used 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)
#Let us chech the aplicability of the model 
#i.e miss classification, Actual patient is malignant 
#i.e cancer patient but predicted is Benien  is 1 
#Actual patient is Benien and predicted as cancer patient is 3
#Hence this model is not acceptable 
############################################################## 
#le tus try to select correct value of k 
acc=[]
#Running KNN algo for k=3 to 50 in the step of 2
#k value selected is odd value 
for i in range(3,50,2):
    #Declare the model 
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    acc.append([train_acc,test_acc])
#if you will see the acc, it has got two accuracy i[0]-train_acc,i[1] - test_acc
#to plot the graph of train_acc and test_acc 
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0]for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[0]for i in acc],'bo-')

#There are 3, 5,7, and 9 possible values where accuracy is goot
#let us check for k=3
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

accuracy_score(pred, y_test)
pd.crosstab(pred, y_test)
#ie. is Classification, Actual patient is Malignangt
#ie cancer patient but predicted is benien is 1
#Actual patient is Benien and predicted as cancer patient is 2
#Hence this model is not acceptable
#for 5 more sinario
#for k = 7 we are getting zero false positive and good accuracy
# Hence k = 7 is appropriate value of k
######==================EDA=================#### 
df = pd.read_csv('wbcd.csv')
df['diagnosis'].value_counts()
#This is a imbalance data set 
#hence we will have to calculate precision recall and then f1 score
