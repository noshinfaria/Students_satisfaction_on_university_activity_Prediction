# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 21:02:05 2021

@author: Noshin
"""
#%%
# Importing the libraries
import function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#!pip install mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


df = pd.read_csv('student_satisfaction.csv')
print(df.isnull().sum())

#%%
# preprocessing


columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','Satisfied']
le = LabelEncoder()
df[columns] = df[columns].apply(le.fit_transform)
print(df.head())
correlation = df.corr()
print(correlation)
sns.heatmap(correlation)

#%%
df1 = df.iloc[:,0:12]
df1['Satisfied'] = pd.Series(df['Satisfied'], index=df1.index)
print(df1.corr())

df2 = df.iloc[:,12:]
print(df2.corr())
sns.heatmap(df2.corr())


#pd.plotting.scatter_matrix(df1.loc[:,'A9':'Satisfied'])
#%%
fig, ax = plt.subplots(3, 4, sharex='col', sharey='row')

m=12
for i in range(3):
    for j in range(4):

        df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
        m+=1
        
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

m=25
for i in range(2):
    for j in range(3):

        df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
        m+=1
 

#%%


class0 =df[['TLA1', 'Satisfied']].loc[df['Satisfied'] == 0]
negative = class0['TLA1']
class1 =df[['TLA1', 'Satisfied']].loc[df['Satisfied'] ==1]
positive = class1['TLA1']

columns = [negative,positive]

fig, ax = plt.subplots()
ax.hist(columns)
plt.xlabel('student satisfection on Interactive and supportive teaching-learning')
plt.ylabel('number of people')
plt.show()


class0 =df[['TLA5', 'Satisfied']].loc[df['Satisfied'] == 0]
negative = class0['TLA5']
class1 =df[['TLA5', 'Satisfied']].loc[df['Satisfied'] ==1]
positive = class1['TLA5']

columns = [negative,positive]

fig, ax = plt.subplots()
ax.hist(columns)
plt.xlabel('student satisfection on Appropriate assessments')
plt.ylabel('number of people')
plt.show()



#%%
X1 = df[['IA1','IA2','IA3','IA4','IA5','IA6','AA1','AA2','AA3','AA4','AA5','AA6','TLA1','TLA2','TLA3','TLA4','TLA5','TLA6','TLA7','Satisfied']]
Y1 = df[['OR']]

x1 = X1.values
y1 = Y1.values

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.30, random_state=40)
print(x1_train.shape); print(y1_train.shape)


#%%
#your own implementation using formula of linear algebra

# - - - - - - - - - - - call multiple_regression function for train data - - - - - - - - - - - - - - - - - - - - - - - - 

y_predict = function.multiple_rgression(x1_train,y1_train)

# - - - - - - - - - - - results - - - - - - - - - - - - - - - - - - - - - - - -
Mean = function.mean(y1_train)
length = len(y1_train)
a = 0
b = 0
for i, j in zip(y1_train, y_predict):
    a += ((i-j)**2)
    b += ((i-Mean)**2)
r2 = 1-(a/b)

c = 0
for i, j in zip(y1_train, y_predict):
    c += (abs(j-i))
mae = c/length

mse = (1/length) * a
print("own method : The model performance for train data : ")
print('R2 score is %.4f'% r2)
print('MSE is %.4f'% mse)
print('MAE is %.4f'% mae)



#%%
#statsmodel OLS method for train data
x = sm.add_constant(x1_train)
model = sm.OLS(y1_train,x1_train)
result = model.fit()
print(" - - - - - - - - - - - -For train data - -  - - - - -  - - - - - - - - - -")
print(result.summary())




#%%
# linear regression using scikit-learn library (without any regularization)
lr = LinearRegression()
lr.fit(x1_train, y1_train)
pred_train_lr= lr.predict(x1_train)

lr_train_mae = mean_absolute_error(y1_train,pred_train_lr)
lr_train_mse = mean_squared_error(y1_train,pred_train_lr)
lr_train_r2 = r2_score(y1_train, pred_train_lr)

print("LinearRegression : The model performance for training set")
print("--------------------------------------")
print('train MAE = %.4f'% lr_train_mae)
print('train MSE = %.4f'% lr_train_mse)
print('train R2 score = %.4f'% lr_train_r2)

pred_test_lr= lr.predict(x1_test)

lr_test_mae = mean_absolute_error(y1_test,pred_test_lr)
lr_test_mse = mean_squared_error(y1_test,pred_test_lr)
lr_test_r2 = r2_score(y1_test, pred_test_lr)

print("LinearRegression : The model performance for testing set")
print("--------------------------------------")
print('test MAE = %.4f'% lr_test_mae)
print('test MSE = %.4f'% lr_test_mse)
print('test R2 score = %.4f'% lr_test_r2)

test = y1_test.flatten()
predict_test = pred_test_lr.flatten()

train = y1_train.flatten()
predict_train = pred_train_lr.flatten()

bins = np.linspace(1, 10, 20)
plt.hist([train, predict_train], bins, label=['y1_train', 'predict_train'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("train data(linear regression)")
plt.show()

bins = np.linspace(1, 10, 20)
plt.hist([test, predict_test], bins, label=['y1_test', 'predict_test'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("test data(linear regression)")
plt.show()

plt.scatter(train,predict_train)
plt.xlabel("y train")  
plt.ylabel("y predict") 
plt.title("train data(linear regression)")
plt.show()

plt.scatter(test,predict_test)
plt.title("test data(linear regression)")  
plt.xlabel("y test")  
plt.ylabel("y predict")
plt.title("test data(linear regression)")
plt.show()

dataset1 = pd.DataFrame({'test': test, 'test_predict': predict_test}, columns=['test', 'test_predict'])
dataset2 = pd.DataFrame({'train': train, 'train_predict': predict_train}, columns=['train', 'train_predict'])
sns.lmplot(x='test', y='test_predict', data=dataset1)
sns.lmplot(x='train', y='train_predict', data=dataset2)


#%%
#lasso regression (L1 regularization)
#without alpha train r2 = 0; test r2 = -0.06
model_lasso = Lasso(alpha=0.01)
model_lasso.fit(x1_train, y1_train) 
pred_train_lasso= model_lasso.predict(x1_train)

la_train_mae = mean_absolute_error(y1_train,pred_train_lasso)
la_train_mse = mean_squared_error(y1_train,pred_train_lasso)
la_train_r2 = r2_score(y1_train, pred_train_lasso)

print("lasso regression : The model performance for training set")
print("--------------------------------------")
print('train MAE = %.4f'% la_train_mae)
print('train MSE = %.4f'% la_train_mse)
print('train R2 score = %.4f'% la_train_r2)

pred_test_lasso= model_lasso.predict(x1_test)
la_test_mae = mean_absolute_error(y1_test,pred_test_lasso)
la_test_mse = mean_squared_error(y1_test,pred_test_lasso)
la_test_r2 = r2_score(y1_test, pred_test_lasso)

print("lasso regression : The model performance for testing set")
print("--------------------------------------")
print('test MAE = %.4f'% la_test_mae)
print('test MSE = %.4f'% la_test_mse)
print('test R2 score = %.4f'% la_test_r2)

test = y1_test.flatten()
predict_test = pred_test_lasso.flatten()
train = y1_train.flatten()
predict_train = pred_train_lasso.flatten()

bins = np.linspace(1, 10, 20)
plt.hist([train, predict_train], bins, label=['y1_train', 'predict_train'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("train data(lasso regression)")
plt.show()

bins = np.linspace(1, 10, 20)
plt.hist([test, predict_test], bins, label=['y1_test', 'predict_test'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("test data(lasso regression)") 
plt.show()

plt.scatter(train,predict_train)
plt.title("train data(lasso regression)")  
plt.xlabel("y train")  
plt.ylabel("y predict")
plt.title("test data(lasso regression)") 
plt.show()

plt.scatter(test,predict_test)
plt.title("test data(lasso regression)")  
plt.xlabel("y test")  
plt.ylabel("y predict")
plt.show()


dataset1 = pd.DataFrame({'test': test, 'test_predict': predict_test}, columns=['test', 'test_predict'])
dataset2 = pd.DataFrame({'train': train, 'train_predict': predict_train}, columns=['train', 'train_predict'])
sns.lmplot(x='test', y='test_predict', data=dataset1)
sns.lmplot(x='train', y='train_predict', data=dataset2)

#%%
#Ridge regression (L2 regularization)
rr = Ridge(alpha=0.1)
rr.fit(x1_train, y1_train) 
pred_train_rr= rr.predict(x1_train)

ri_train_mae = mean_absolute_error(y1_train,pred_train_rr)
ri_train_mse = mean_squared_error(y1_train,pred_train_rr)
ri_train_r2 = r2_score(y1_train, pred_train_rr)

print("Ridge regression : The model performance for training set")
print("--------------------------------------")
print('train MAE = %.4f'% ri_train_mae)
print('train MSE = %.4f'% ri_train_mse)
print('train R2 score = %.4f'% ri_train_r2)

pred_test_rr= rr.predict(x1_test)

ri_test_mae = mean_absolute_error(y1_test,pred_test_rr)
ri_test_mse = mean_squared_error(y1_test,pred_test_rr)
ri_test_r2 = r2_score(y1_test, pred_test_rr)

print("Ridge regression : The model performance for testing set")
print("--------------------------------------")
print('test MAE = %.4f'% ri_test_mae)
print('test MSE = %.4f'% ri_test_mse)
print('test R2 score = %.4f'% ri_test_r2)

train = y1_train.flatten()
predict_train = pred_train_rr.flatten()

test = y1_test.flatten()
predict_test = pred_test_rr.flatten()

bins = np.linspace(1, 6, 10)
plt.hist([train, predict_train], bins, label=['y_train', 'predict'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("train data(Ridge regression)")
plt.show()

bins = np.linspace(1, 8, 10)
plt.hist([test, predict_test], bins, label=['y_test', 'predict'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("test data(Ridge regression)") 
plt.show()

plt.scatter(train,predict_train)
plt.title("train data(Ridge regression)")  
plt.xlabel("y train")  
plt.ylabel("y predict") 
plt.show()

plt.scatter(test,predict_test)
plt.title("test data(Ridge regression)")  
plt.xlabel("y test")  
plt.ylabel("y predict")
plt.show()


dataset1 = pd.DataFrame({'test': test, 'test_predict': predict_test}, columns=['test', 'test_predict'])
dataset2 = pd.DataFrame({'train': train, 'train_predict': predict_train}, columns=['train', 'train_predict'])
sns.lmplot(x='test', y='test_predict', data=dataset1)
sns.lmplot(x='train', y='train_predict', data=dataset2)

#%%
#polynomial regression using scikit-learn library 

#bad result for increasing degree


poly = PolynomialFeatures(degree =2)
x_train_poly= poly.fit_transform(x1_train)
x_test_poly= poly.fit_transform(x1_test)

model1 = LinearRegression()
model1.fit(x_train_poly, y1_train)

predict_train_poly = model1.predict(x_train_poly)
poly_train_mae = mean_absolute_error(y1_train,predict_train_poly)
poly_train_mse = mean_squared_error(y1_train,predict_train_poly)
poly_train_r2 = r2_score(y1_train, predict_train_poly)

print("polynomial regression : The model performance for training set")
print("--------------------------------------")
print('train MAE = %.4f'% poly_train_mae)
print('train MSE = %.4f'% poly_train_mse)
print('train R2 score = %.4f'% poly_train_r2)

predict_test_poly = model1.predict(x_test_poly)
poly_test_mae = mean_absolute_error(y1_test,predict_test_poly)
poly_test_mse = mean_squared_error(y1_test,predict_test_poly)
poly_test_r2 = r2_score(y1_test, predict_test_poly)

print("polynomial regression : The model performance for testing set")
print("--------------------------------------")
print('test MAE = %.4f'% poly_test_mae)
print('test MSE = %.4f'% poly_test_mse)
print('test R2 score = %.4f'% poly_test_r2)

train = y1_train.flatten()
predict_train = predict_train_poly.flatten()

test = y1_test.flatten()
predict_test = predict_test_poly.flatten()

plt.hist([train, predict_train], bins, label=['y_train', 'predict'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("train data(polynomial regression)")
plt.show()

bins = np.linspace(1, 8, 10)
plt.hist([test, predict_test], bins, label=['y_test', 'predict'])
plt.legend(loc='upper right')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.title("test data(polynomial regression)") 
plt.show()

plt.scatter(y1_train,predict_train_poly)
plt.title("train data(Polynomial)")  
plt.xlabel("y train")  
plt.ylabel("y predict") 
plt.show()

plt.scatter(y1_test,predict_test_poly)
plt.title("test data(Polynomial)")  
plt.xlabel("y test")  
plt.ylabel("y predict")
plt.show()

dataset1 = pd.DataFrame({'test': test, 'test_predict': predict_test}, columns=['test', 'test_predict'])
dataset2 = pd.DataFrame({'train': train, 'train_predict': predict_train}, columns=['train', 'train_predict'])
sns.lmplot(x='test', y='test_predict', data=dataset1)
sns.lmplot(x='train', y='train_predict', data=dataset2)


#%%

X2 = df.drop('Satisfied',axis=1)
Y2 = df['Satisfied']

x2 = X2.values
y2 = Y2.values
print(type(y2))
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.30, random_state=40)
print(x2_train.shape); print(x2_test.shape)




#%%
#LOGISTIC REGRESSION using grid search parameter
logic = LogisticRegression()

grid = [
        {'penalty' : ['l1','l2','elasticnet', 'none'],
         'C' : np.logspace(-4, 2, 20),
         'solver' : ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
         'max_iter': [180,200]
         }
        ]

clf = GridSearchCV(logic, param_grid = grid,cv = 3,verbose= True, n_jobs = -1)
best_clf = clf.fit(x2_train, y2_train)
#%%
print(best_clf.best_estimator_)
print(best_clf.score(x2_test,y2_test))


#%%
#
logic_train_pred = best_clf.predict(x2_train)
lo_accuracy_train = metrics.accuracy_score(y2_train, logic_train_pred)

logic_test_pred = best_clf.predict(x2_test)
lo_accuracy_test = metrics.accuracy_score(y2_test, logic_test_pred)

print('Logistic regression : ')
print('train accuracy = %.4f'% lo_accuracy_train)
print('test accuracy = %.4f'% lo_accuracy_test)

lo_cm_train = confusion_matrix(y2_train, logic_train_pred)
print('confusion matrix of train data : ' )
print(lo_cm_train)

lo_cm_test = confusion_matrix(y2_test, logic_test_pred)
print('confusion matrix of test data : ' )
print(lo_cm_test)

sns.heatmap(lo_cm_train, annot=True, cmap="vlag_r")
plt.title("Confusion Metrix for train dataset(logistic regression)", fontsize=14, fontname="Helvetica", y=1.03);
#%%
sns.heatmap(lo_cm_test, annot=True, cmap="vlag_r")
plt.title("Confusion Metrix for test dataset(logistic regression)", fontsize=14, fontname="Helvetica", y=1.03);
#%%
lo_report = classification_report(y2_test,logic_test_pred)
print(lo_report)

test = y2_test.flatten()
predict_test = logic_test_pred.flatten()
train = y2_train.flatten()
predict_train = logic_train_pred.flatten()


plt.hist([train, predict_train], bins=3,alpha=1, label=['y_train', 'predict'])
plt.legend(loc='upper left')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.xticks([-1,0,1,2])
plt.show()

plt.hist([test, predict_test], bins=3,alpha=1, label=['y_test', 'predict'])
plt.legend(loc='upper left')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.xticks([-1,0,1,2])
plt.show()




#%%
#svm from c = 9 to 12 = the sweet spot
clf = svm.SVC(kernel='linear',C=9)
clf.fit(x2_train, y2_train)

svm_train_pred = clf.predict(x2_train)
svm_a_train = metrics.accuracy_score(y2_train, svm_train_pred)

svm_test_pred = clf.predict(x2_test)
svm_a_test = metrics.accuracy_score(y2_test, svm_test_pred)

print('svm = train accuracy = %.4f'% svm_a_train)
print('svm = test accuracy = %.4f'% svm_a_test)

svm_c_train = confusion_matrix(y2_train, svm_train_pred)
print('svm = confusion matrix of train data : ' )
print(svm_c_train)

svm_c_test = confusion_matrix(y2_test, svm_test_pred)
print('svm = confusion matrix of test data : ' )
print(svm_c_test)

sns.heatmap(svm_c_train, annot=True, cmap="vlag_r")
plt.title("Confusion Metrix for train dataset(SVM)", fontsize=14, y=1.03);
#%%
sns.heatmap(svm_c_test, annot=True, cmap="vlag_r")
plt.title("Confusion Metrix for test dataset(SVM)", fontsize=14, y=1.03);
#%%
svm_report = classification_report(y2_test,svm_test_pred)
print(svm_report)

test = y2_test.flatten()
predict_test = svm_test_pred.flatten()
train = y2_train.flatten()
predict_train = svm_train_pred.flatten()

plt.hist([train, predict_train], bins=3,alpha=1, label=['y_train', 'predict'])
plt.legend(loc='upper left')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.xticks([-1,0,1,2])
plt.show()

plt.hist([test, predict_test], bins=3,alpha=1, label=['y_test', 'predict'])
plt.legend(loc='upper left')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.xticks([-1,0,1,2])
plt.show()

#%%
#using Backward Feature Elimination K-Nearest Neighbours (KNN algorithm) 
X3 = X2
Y3 = Y2
classifier = KNeighborsClassifier(n_neighbors = 10, p=1, weights='uniform')
sfs1 = sfs(classifier, k_features=16, forward=False, verbose=1, scoring='accuracy', n_jobs=-1)
sfs1.fit(X3, Y3)


feat_names = list(sfs1.k_feature_names_)
print(feat_names)
X3 = df[feat_names]

x3 = X3.values
y3 = Y3.values

x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.30, random_state=40)
print(x3_train.shape); print(x3_test.shape)
#%%
classifier.fit(x3_train, y3_train)
k_train_pred = classifier.predict(x3_train)
accuracy_train = metrics.accuracy_score(y3_train, k_train_pred)

k_test_pred = classifier.predict(x3_test)
accuracy_test = metrics.accuracy_score(y3_test,k_test_pred)

print('Backward Feature Elimination K-Nearest Neighbours : ')
print('train accuracy = %.4f'% accuracy_train)
print('test accuracy = %.4f'% accuracy_test)

c_train = confusion_matrix(y3_train, k_train_pred)
print('confusion matrix of train data : ' )
print(c_train)

c_test = confusion_matrix(y3_test, k_test_pred)
print('confusion matrix of test data : ' )
print(c_test)

sns.heatmap(c_train, annot=True, cmap="vlag_r")
plt.title("Confusion Metrix for train dataset(SVM)", fontsize=14, fontname="Helvetica", y=1.03);
#%%
sns.heatmap(c_test, annot=True, cmap="vlag_r")
plt.title("Confusion Metrix for test dataset(SVM)", fontsize=14, fontname="Helvetica", y=1.03);

#%%
report = classification_report(y3_test,k_test_pred)
print(report)

test = y3_test.flatten()
predict_test = k_train_pred.flatten()
train = y3_train.flatten()
predict_train = k_test_pred.flatten()

plt.hist([train, k_train_pred], bins=3,alpha=1, label=['y_train', 'predict'])
plt.legend(loc='upper left')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.xticks([-1,0,1,2])
plt.show()

plt.hist([test, k_test_pred], bins=3,alpha=1, label=['y_test', 'predict'])
plt.legend(loc='upper left')
plt.xlabel('OR')
plt.ylabel('number of student')
plt.xticks([-1,0,1,2])
plt.show()



#%%
plt.style.use("seaborn")

x = ["Linear", 
     "Lasso", 
     "Ridge",
     "Polynomial",
     "Logistic",
     "SVM",
     "KNN"]

y = [lr_train_r2, 
     la_train_r2, 
     ri_train_r2,
     poly_train_r2,
     lo_accuracy_train,
     svm_a_train,
     accuracy_train]

fig, ax = plt.subplots(figsize=(15,8))
sns.barplot(x=x,y=y, palette="hot");
plt.ylabel("Model Accuracy")
plt.xticks(rotation=40)
plt.title("Train Model Comparison - Model Accuracy", fontsize=20, fontname="Helvetica", y=1.03);

#%%
plt.style.use("seaborn")

x = ["Linear", 
     "Lasso", 
     "Ridge",
     "Polynomial",
     "Logistic",
     "SVM",
     "KNN"]

y = [lr_test_r2, 
     la_test_r2, 
     ri_test_r2,
     poly_test_r2,
     lo_accuracy_test,
     svm_a_test,
     accuracy_test]

fig, ax = plt.subplots(figsize=(15,8))
sns.barplot(x=x,y=y, palette="hot");
plt.ylabel("Model Accuracy")
plt.xticks(rotation=40)
plt.title("Test Comparison - Model Accuracy", fontsize=20, fontname="Helvetica", y=1.03);


#%%
test_df = pd.read_csv('student_satisfaction_for_testing.csv')
print(test_df.isnull().sum())
#%%

columns = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']
l = LabelEncoder()
test_df[columns] = test_df[columns].apply(l.fit_transform)
print(test_df.head())

final_pred = best_clf.predict(test_df)

#%% no 4 roc curve for logistic regression since it gives best value

logistic_fpr,logistic_tpr,threshold = roc_curve(y2_test, logic_test_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)


plt.figure(figsize=(6,5),dpi=100)
plt.plot(logistic_fpr,logistic_tpr,label= 'Logistic(auc= %0.3f)'% auc_logistic)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')

#%% no 5
print (df[[ 'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']].corrwith(df['Satisfied']))

print ('most correlated satisfied column',max(df[[ 'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']].corrwith(df['Satisfied'])))


