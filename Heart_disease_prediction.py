#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.metrics import confusion_matrix

#Read the data file
df = pd.read_csv("prediction_dataset.csv")
#print(df.head())

#Data Preprocessing
#print(df.dtypes)
#print(df.isnull().sum())
#print(df.describe())

#Removing null rows & replacing them with median
df['male'] = df['male'].fillna(df['male'].median())
df['age'] = df['age'].fillna(df['age'].median())
df['education'] = df['education'].fillna(df['education'].median())
df['currentSmoker'] = df['currentSmoker'].fillna(df['currentSmoker'].median())
df['cigsPerDay'] = df['cigsPerDay'].fillna(df['cigsPerDay'].median())
df['BPMeds'] = df['BPMeds'].fillna(df['BPMeds'].median())
df['prevalentStroke'] = df['prevalentStroke'].fillna(df['prevalentStroke'].median())
df['prevalentHyp'] = df['prevalentHyp'].fillna(df['prevalentHyp'].median())
df['diabetes'] = df['diabetes'].fillna(df['diabetes'].median())
df['totChol'] = df['totChol'].fillna(df['totChol'].median())
df['sysBP'] = df['sysBP'].fillna(df['sysBP'].median())
df['diaBP'] = df['diaBP'].fillna(df['diaBP'].median())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['heartRate'] = df['heartRate'].fillna(df['heartRate'].median())
df['glucose'] = df['glucose'].fillna(df['glucose'].median())

#print(df.isnull().sum())
#Drop unwanted columns
df = df.drop("education",axis = 1)
df = df.drop("prevalentStroke",axis = 1)
df = df.drop("prevalentHyp",axis = 1)
#Data Splitting

#print(df.columns)
X = df.drop("TenYearCHD",axis = 1)
Y = df['TenYearCHD']



#print(X.columns)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1400)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#4.Train the Model
my_model = KNeighborsClassifier(n_neighbors = 4)
result = my_model.fit(X_train,Y_train)
#5.Test the model
prediction = result.predict(X_test)
#print(prediction)
print("Accuracy:",metrics.accuracy_score(Y_test,prediction))
		


predict_list = []
while len(predict_list) == 0:
	male = input("Enter your gender(M/F):")
	if(male == 'M'):
		predict_list.insert(0,1)
	elif male == 'F':
		predict_list.insert(0,0)

age = int(input("Enter your age:"))
predict_list.insert(1,age)

while len(predict_list) == 2:
	currSmoker = input("Are you a smoker currently?(Type Y for yes and N for no):")
	if(currSmoker == 'Y'):
		predict_list.insert(2,1)
		cigsPerDay = int(input("Enter total number of cigarattes intake per day:"))
		predict_list.insert(3,cigsPerDay)
	elif currSmoker == 'N':
		predict_list.insert(2,0)
		predict_list.insert(3,0)



#print(len(predict_list))

while len(predict_list) == 4:
	BP = input("Do you suffer from blood pressure(Type Y for yes and N for No):")
	if(BP == 'Y'):
		predict_list.insert(4,1)
	elif BP == 'N':
		predict_list.insert(4,0)

while len(predict_list) == 5:
	diabetes = input("Do you suffer from diabetes(Type Y for yes and N for No):")
	if(diabetes == 'Y'):
		predict_list.insert(5,1)
	elif diabetes == 'N':
		predict_list.insert(5,0)		

cholestrol = int(input("Enter your total cholestrol level:"))
predict_list.insert(6,cholestrol)

systolic = int(input("Enter systolic blood pressure:"))
predict_list.insert(7,systolic)

diastolic = int(input("Enter diastolic blood pressure:"))
predict_list.insert(8,diastolic)

BMI = float(input("Enter your current BMI(Body Mass Index):"))
predict_list.insert(9,BMI)

heart_rate = int(input("Enter your heart rate:"))
predict_list.insert(10,heart_rate)

glucose_level = int(input("Enter your glucose level:"))
predict_list.insert(11,glucose_level)

#print(predict_list)
new_predict = result.predict([predict_list])
print(new_predict)

if new_predict == 0:
	print("Congratulations!You are not at risk of getting heart diseases atleast for next 10 years")
else:
	print("There is a risk of getting heart diseases in the next 10 years")



