import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



df=pd.read_csv("heart.csv")
df.head()
print(df.target.value_counts())

sns.countplot(x="target",data=df,palette="bwr")
#plt.show()
CountNoDisease=len(df[df.target==0])
CountHaveDisease=len(df[df.target==1])
print("percentage of no disease: {:.2f}%".format((CountNoDisease)/(len(df.target))*100))
print("percentage of have disease: {:.2f}%".format((CountHaveDisease)/(len(df.target))*100))


sns.countplot(x='sex',data=df,palette="bwr")
plt.xlabel("sex 0=female,1=male")
#plt.show()
CountMale=(len(df[df.sex == 1]))
CountFemale=(len(df[df.sex == 0]))

print("Percentage of male: {:.2f}%".format((CountMale)/(len(df.sex))*100))
print("Percentage of Female :{:.2f}%".format((CountFemale)/(len(df.sex))*100))

#how to get the percentage of male/female having disease
print(df.groupby(by='target').mean())

pd.crosstab(df.sex,df.target).plot(kind="bar")
#plt.show()

y=df.target.values

x_data=df.drop(['target'],axis=1)

# this step is called normalization and we do normalization so that the scale can be same for all the dependent variables so the changes will be equal
# like area is in 1000 sq. and numberof rooms is in 1-4 so if area increases by 1 it doesnt make that much difference but when room increase by 1 it makes
# a huge difference in the output so by normalization we are using x-xmin/xmax-xmin

x=(x_data-np.min(x_data)/(np.max(x_data)-(np.min(x_data))))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=0)


# in ths next step we transpose the data
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T


#LOGISTIC REGRESSION

lr=LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("\n\n\n Logistic Regression \n Percentage of Accuracy is :{:.2f}%".format(lr.score(x_test.T,y_test.T)*100))

#KNN CLASSIFIER

from sklearn.neighbors import KNeighborsClassifier
# this is to find the best value of k
# scoreList=[]
# for i in range(1,20):
#     knn2=KNeighborsClassifier(n_neighbors=i)
#     knn2.fit(x_train.T,y_train.T)
#     scoreList.append(knn2.score(x_test.T,y_test.T))
# print(scoreList)
# print("Max score is {:.2f}%".format((max(scoreList))*100))
#


knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train.T,y_train.T)
prediction=knn.predict(x_test.T)
print(knn.score(x_test.T,y_test.T)*100)
print("\n\n KNN Classifier \n Percentage of accuracy is {:.2f}%".format(knn.score(x_test.T,y_test.T)*100))

# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)
print("\n\n Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test.T,y_test.T)*100))


#DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)
print("\n\n Decision Tree Test Accuracy {:.2f}%".format(dtc.score(x_test.T, y_test.T)*100))

#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)
print("\n\n Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test.T,y_test.T)*100))