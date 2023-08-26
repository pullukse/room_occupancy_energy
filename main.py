import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Users/seray.pulluk/Occupancy_datatest2.txt')
df_2 = pd.read_csv('Users/seray.pulluk/Occupancy_datatest.txt')

df_new = pd.concat([df, df_2])


#EDA & Viz

# Box Plot for Light
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_new, x='Light')
plt.xlabel('Light')
plt.title('Box Plot for Light')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_new.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Box Plot for CO2 by Occupancy
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_new, x='Occupancy', y='CO2')
plt.xlabel('Occupancy')
plt.ylabel('CO2')
plt.title('Box Plot for CO2 by Occupancy')
plt.show()

# Histogram for Humidity
plt.figure(figsize=(8, 6))
plt.hist(df_new['Humidity'], bins=10, edgecolor='black')
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Histogram for Humidity')
plt.show()

#Split date-time
df_new['Dates'] = pd.to_datetime(df_new['date']).dt.date
df_new['Time'] = pd.to_datetime(df_new['date']).dt.time
df_new = df_new.drop('date', axis=1)

#Round the nums
df_new[['Temperature','Humidity','Light','CO2']] = df_new[['Temperature','Humidity','Light','CO2']].round(2)

#DROPPING DATE + TIME FOR ML MODELS
df_new1 = df_new.drop(['Dates', 'Time'], axis=1)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 

#LOG REG
X_train, X_test, y_train, y_test = train_test_split(df_new1.drop('Occupancy', axis = 1), df_new1['Occupancy'], test_size=0.3, random_state=42, stratify = df_new1['Occupancy'])

lr = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = lr.predict(X_test)

roc_auc_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix

#RANDOM FOREST
rfc=RandomForestClassifier(random_state=42)

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

#Hyper. tuning
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

rfc=RandomForestClassifier(criterion = 'entropy', max_depth = 8, max_features = 'auto', n_estimators = 200)
rfc.fit(X_train, y_train)

pred=rfc.predict(X_test)

print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))

print(classification_report(y_test, pred))

cnf_matrix = confusion_matrix(y_test, pred)
cnf_matrix




