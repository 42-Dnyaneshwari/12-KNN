#Assignment on KNN
#on dataset Zoo
# Importing necessary libraries
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/KNN/Zoo.csv", encoding="ISO-8859-1")

# Data cleaning
df.fillna(value='missing', inplace=True)

# EDA
df.info()
df.describe()

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['fins'], kde=True)
plt.title('Histogram of fins')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Na', data=df)
plt.title('Boxplot of Na')
plt.xlabel('Age')
plt.ylabel('Values')
plt.show()

#Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(df['fins'])
plt.title('scatterplot of fins')
plt.xlabel('fins')
plt.ylabel('Values')
plt.show()

# Model Building

df.describe()
df['fins'] = np.where(df['fins'] == 'B', 'Beniegn', df['Na'])
df['fins'] = np.where(df['fins'] == 'M', 'Malignant', df['Na'])
########################################################################33 
#0th Column is patient ID let us drop it 
df = df.iloc[:,1:32]
#########################################################################3
#Normalisation 
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())
df_n = norm_func(df.iloc[:,1:32])
#becaue now 0th column is output or lable it is not considered hence l
####################################################################### 
X = np.array(df_n.iloc[:,:])
y = np.array(df['fins'])
##############################################################33 
#Now let us split the data into training and testing 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)

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

######===================================#### 
