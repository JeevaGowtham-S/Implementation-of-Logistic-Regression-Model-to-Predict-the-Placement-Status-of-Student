# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student->

## AIM :
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## PROGRAM :
#### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#### Developed by : JEEVAGOWTHAM S
#### RegisterNumber : 212222230053
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## OUTPUT:

### Placement_data
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/8cc95c47-7ec3-4b4f-890a-55272e905819)


### Salary_data
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/36da1533-e7d7-4b56-b125-a5ccf6b9efa3)


### ISNULL()
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/ed5e93f2-40b2-4cf9-a8b9-48cdb28d002f)



### DUPLICATED()
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/05e53e19-9659-4fce-8bfd-44a88b31bb07)


### Print Data
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/df46a9ff-bcd4-4785-a7ad-33ab848e848f)


### iloc[:,:-1]
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/2b588b75-5594-46f7-aafd-b3f471f9bd6b)


### Data_Status
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/6c3fa8c6-e710-479c-a158-6a34aa51c87f)


### Y_Prediction array:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/fa84ff65-f2ca-4666-927c-c703f3cc1722)


### Accuray value:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/b887da52-6250-4021-99a7-1a37117a6f30)


### Confusion Array:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/1ee1084f-d4ef-4673-b8e4-18e7c05ae8a1)


### Classification report:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/1305a565-45cb-48fa-954b-bbb0c48602c7)

### Prediction of LR:
![image](https://github.com/JeevaGowtham-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118042624/efefb1d3-9da7-47d0-9a6f-354e89b9c106)



## RESULT :
Thus,the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
