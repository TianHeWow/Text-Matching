#! /usr/bin/python
# -*- coding: utf-8 -*-
from utils import Processing, Loader
from featureExtraction import Not, WordMatchShare,\
     Length, NgramJaccardCoef, long_common_sequence,\
     fuzz_partial_token_set_ratio, fuzz_partial_token_sort_ratio,\
     TFIDFSpanish, Label

if __name__ == '__main__':
    Model_Flag = 3
    config_fp = './featwheel.conf'

    if Model_Flag == 1:
        data_pt = '../data/RowData/cikm_21400.txt'
        save_pt = '../data/PreProcessingData/cikm_spanish_train.csv'
        Processing().excute_csv(data_pt, save_pt)

    if Model_Flag == 2:
        data_fp = '../data/PreProcessingData/cikm_spanish_train.csv'
        feature_version = 0
        Not(config_fp).extract(data_fp=data_fp, feature_version=0)
        WordMatchShare(config_fp).extract(data_fp=data_fp, feature_version=0)
        Length(config_fp).extract(data_fp=data_fp, feature_version=0)
        NgramJaccardCoef(config_fp).extract(data_fp=data_fp, feature_version=0)
        long_common_sequence(config_fp).extract(data_fp=data_fp, feature_version=0)
        fuzz_partial_token_set_ratio(config_fp).extract(data_fp=data_fp, feature_version=0)
        fuzz_partial_token_sort_ratio(config_fp).extract(data_fp=data_fp, feature_version=0)
        TFIDFSpanish(config_fp, data_fp).extract(data_fp=data_fp, feature_version=0)
        Label(config_fp).extract(data_fp=data_fp, feature_version=0)

    if Model_Flag == 3:
        DataSpanishSenPair = Loader(config_fp).loadAllData()
        print(DataSpanishSenPair.keys())
        print(DataSpanishSenPair['Labels'])
        print(len(DataSpanishSenPair['Labels']))
        print(sum(DataSpanishSenPair['Labels']))
        
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
data = DataSpanishSenPair['Features'].assign(L = DataSpanishSenPair['Labels'])
data = data.drop(['Length_0_0', 'Length_0_1', 'Length_0_2', 'Length_0_3', 'Not_0_0', 'Not_0_1',
       'Not_0_2', 'Not_0_3', 'TFIDFSpanish_0_0',
       'TFIDFSpanish_0_1', 'TFIDFSpanish_0_2', 'TFIDFSpanish_0_3'], axis = 1)
data.to_csv("spanishMatching_data")
correlation = data.corr()
##draw correlation matrix
sns.heatmap(correlation, 
        xticklabels=correlation.columns,
        yticklabels=correlation.columns)
## delete columns with high correlation
corr_matrix = correlation.abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print("We will delete columns having high correlation with other columns: ", to_drop)
data = data.drop(to_drop, axis = 1)
corr_drop = data.corr()
## draw correlation matrix
sns.heatmap(corr_drop, 
        xticklabels=corr_drop.columns,
        yticklabels=corr_drop.columns)

X = data.drop(["L","ARCI_deepModel_0","ARCII_deepModel_0"], axis = 1)
Y = data["L"].values
##The number of features are small, and we also wants to get some information 
## about which feature might be useful. We do not decompose features.
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0)
plt.hist(y_train)
## explore skewness
plt.title('Histogram for labels')
plt.xlabel('Class')
plt.ylabel('frequency')
plt.show()
print("in the train set, we have ", y_train.sum(), "Class 1, ", len(y_train) - y_train.sum(), "Class 0. The skewness is not severe")
## logistic regression
### The first model we want to try is logistic regression. Logistic regression
### is simple and it can give us a test of how our features work.
### scale data set at first
from sklearn import preprocessing
x_train_s = preprocessing.scale(x_train)
x_test_s = preprocessing.scale(x_test)
import statsmodels.discrete.discrete_model as sm
logit2=sm.Logit(y_train, x_train_s).fit(maxiter=200)
print(logit2.summary())
print(logit2.pvalues)
[(e1,e2) for (e1, e2) in zip(logit2.pvalues, x_train.columns)]
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.get_params()
param = {'C' : [0.1, 1, 5, 50, 100]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(lor, param_grid = param, cv = 10)
grid_search.fit(x_train_s, y_train)
lor_cv = grid_search.cv_results_
lor_cv_score_mean = lor_cv['mean_test_score']
lor_cv_score_sd = lor_cv['std_test_score']
lor_cv_c = ["0.1", "1", "5", "50", "100"]
plt.plot(lor_cv_c, lor_cv_score_mean, color = 'blue')
plt.errorbar(lor_cv_c, lor_cv_score_mean, yerr = 2*lor_cv_score_sd, color = 'blue', ecolor='red')
## we choose the the smallest c, so our model has less variace 
## which is more suitable for general situation
best_lor = LogisticRegression(C = 0.1).fit(x_train_s, y_train)
best_lor_acc = best_lor.score(x_test_s, y_test)
lor_re = best_lor.predict_proba(x_test_s)
## loss function value
import math
z = 0
for i in range(len(y_test)):
    z = z + y_test[i]*math.log(lor_re[i][1]) + (1-y_test[i])*math.log(lor_re[i][0])
log_loss_lr = -(z/len(y_test))
print("The log loss for logistic regression is ", log_loss_lr)

## Random forest classfication
### random forest has three parameters for tune. Since this modle is designed
### avoid overfitting, so we do not consider accuracy VS complexity parameters 
### here 
from sklearn.ensemble import RandomForestClassifier
r_f = RandomForestClassifier(random_state = 1)
r_f.get_params()
param = { 
    'n_estimators': [500, 1000, 1300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [round(len(x_train.columns)/6),round(len(x_train.columns)/3),round(len(x_train.columns)/2),round(len(x_train.columns)/1)]
}
grid_search_rf = GridSearchCV(r_f, param_grid = param, cv = 5, n_jobs = -1)
grid_search_rf.fit(x_train, y_train)
grid_search_rf.best_params_
best_rf = grid_search_rf.best_estimator_
rf_score = best_rf.score(x_test, y_test)
rf_pre = best_rf.predict_proba(x_test)
rf_pre = pd.DataFrame(rf_pre)
### Change 0 to not equal to 0 to caculate log loss
rf_pre.loc[rf_pre[0] == 0, [0]] = 0.000000000001
## loss function value
z_rf = 0
for i in range(len(y_test)):
    z_rf = z_rf + y_test[i]*math.log(rf_pre.iloc[[i]][1]) + (1-y_test[i])*math.log(rf_pre.iloc[[i]][0])
log_loss_rf = -(z_rf/len(y_test))
print("The log loss for random forest is ", log_loss_rf)


## Support vetor machine classifiction
from sklearn.svm import SVC
svm = SVC()
svm.get_params()
param = {
        'gamma': [0.05, 0.01, 0.001, 0.0001, 0.00001],
        'C': [0.001, 0.1, 1, 10, 100, 1000]}
grid_search_svm = GridSearchCV(svm, param_grid = param, cv = 5, n_jobs = -1)
grid_search_svm.fit(x_train_s, y_train)
svm_cv_param = grid_search_svm.cv_results_['params']
svm_cv_socre_mean = grid_search_svm.cv_results_['mean_test_score']
svm_cv_socre_std = grid_search_svm.cv_results_['std_test_score']
plt.plot(range(len(svm_cv_param)), svm_cv_socre_mean, color = 'blue')
plt.errorbar(range(len(svm_cv_param)), svm_cv_socre_mean, yerr = 2*svm_cv_socre_std, color = 'blue', ecolor='red')
print ("the best parameter is " , svm_cv_param[25])
svm_best = SVC(gamma = 0.05, C = 1000, probability = True).fit(x_train_s, y_train)
svm_score = svm_best.score(x_test_s, y_test)
svm_pre = svm_best.predict_proba(x_test_s)
#rf_pre = pd.DataFrame(rf_pre)
### Change 0 to not equal to 0 to caculate log loss
#rf_pre.loc[rf_pre[0] == 0, [0]] = 0.000000000001
## loss function value
z_svm = 0
for i in range(len(y_test)):
    z_svm = z_svm + y_test[i]*math.log(svm_pre[i][1]) + (1-y_test[i])*math.log(svm_pre[i][0])
log_loss_svm = -(z_svm/len(y_test))
print("The log loss for support vector machine is ", log_loss_svm)




# Add deeplearning features 
data = DataSpanishSenPair['Features'].assign(L = DataSpanishSenPair['Labels'])
data = data.drop(['Length_0_0', 'Length_0_1', 'Length_0_2', 'Length_0_3', 'Not_0_0', 'Not_0_1',
       'Not_0_2', 'Not_0_3', 'TFIDFSpanish_0_0',
       'TFIDFSpanish_0_1', 'TFIDFSpanish_0_2', 'TFIDFSpanish_0_3'], axis = 1)
correlation = data.corr()
##draw correlation matrix
sns.heatmap(correlation, 
        xticklabels=correlation.columns,
        yticklabels=correlation.columns)
## delete columns with high correlation
corr_matrix = correlation.abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print("We will delete columns having high correlation with other columns: ", to_drop)
data = data.drop(to_drop, axis = 1)
corr_drop = data.corr()
## draw correlation matrix
sns.heatmap(corr_drop, 
        xticklabels=corr_drop.columns,
        yticklabels=corr_drop.columns)

X = data.drop(["L"], axis = 1)
Y = data["L"].values
##The number of features are small, and we also wants to get some information 
## about which feature might be useful. We do not decompose features.
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 0)
## explore skewness
plt.hist(y_train)
plt.title('Histogram for labels')
plt.xlabel('Class')
plt.ylabel('frequency')
plt.show()
print("in the train set, we have ", y_train.sum(), "Class 1, ", len(y_train) - y_train.sum(), "Class 0. The skewness is not severe")
## logistic regression
### The first model we want to try is logistic regression. Logistic regression
### is simple and it can give us a test of how our features work.
### scale data set at first
x_train_s = preprocessing.scale(x_train)
x_test_s = preprocessing.scale(x_test)
logit2_d=sm.Logit(y_train, x_train_s).fit(maxiter=200)
print(logit2_d.summary())
print(logit2_d.pvalues)
lor_d = LogisticRegression()
lor_d.get_params()
param = {'C' : [0.1, 1, 5, 50, 100]}
grid_search = GridSearchCV(lor_d, param_grid = param, cv = 10)
grid_search.fit(x_train_s, y_train)
lor_cv_d = grid_search.cv_results_
lor_cv_d_score_mean = lor_cv_d['mean_test_score']
lor_cv_d_score_sd = lor_cv_d['std_test_score']
lor_cv_c_d = ["0.1", "1", "5", "50", "100"]
plt.plot(lor_cv_c_d, lor_cv_d_score_mean, color = 'blue')
plt.title('Logistic regression')
plt.errorbar(lor_cv_c_d, lor_cv_d_score_mean, yerr = 2*lor_cv_d_score_sd, color = 'blue', ecolor='red')
plt.title('Logistic regression')
## we choose the the smallest c, so our model has less variace 
## which is more suitable for general situation
best_lor_d = LogisticRegression(C = 0.1).fit(x_train_s, y_train)
best_lor_d_acc = best_lor_d.score(x_test_s, y_test)
lor_re_d = best_lor_d.predict_proba(x_test_s)
## loss function value
z_d = 0
for i in range(len(y_test)):
    z_d = z_d + y_test[i]*math.log(lor_re_d[i][1]) + (1-y_test[i])*math.log(lor_re_d[i][0])
log_loss_lr_d = -(z_d/len(y_test))
print("The log loss for logistic regression with deep learning feature is ", log_loss_lr_d)

## Random forest classfication
### random forest has three parameters for tune. Since this modle is designed
### avoid overfitting, so we do not consider accuracy VS complexity parameters 
### here 
r_f_d = RandomForestClassifier(random_state = 1)
r_f_d.get_params()
param = { 
    'n_estimators': [500, 1000, 1500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [round(len(x_train.columns)/6),round(len(x_train.columns)/3),round(len(x_train.columns)/2),round(len(x_train.columns)/1)]
}
grid_search_rf_d = GridSearchCV(r_f_d, param_grid = param, cv = 5, n_jobs = -1)
grid_search_rf_d.fit(x_train, y_train)
grid_search_rf_d.best_params_
best_rf_d = grid_search_rf_d.best_estimator_
rf_score_d = best_rf_d.score(x_test, y_test)
rf_pre_d = best_rf_d.predict_proba(x_test)
rf_pre_d = pd.DataFrame(rf_pre_d)
### Change 0 to not equal to 0 to caculate log loss
rf_pre_d.loc[rf_pre_d[0] == 0, [0]] = 0.000000000001
rf_pre_d.loc[rf_pre_d[0] == 1, [0]] = 0.999999999999
## loss function value
z_rf_d = 0
for i in range(len(y_test)):
    z_rf_d = z_rf_d + y_test[i]*math.log(1 - rf_pre_d.iloc[[i]][0]) + (1-y_test[i])*math.log(rf_pre_d.iloc[[i]][0])
log_loss_rf_d = -(z_rf_d/len(y_test))
print("The log loss for random forest with deep learning feature is ", log_loss_rf_d)


## Support vetor machine classifiction
svm_d = SVC()
svm_d.get_params()
param = {
        'gamma': [0.05, 0.01, 0.001, 0.0001, 0.00001],
        'C': [0.001, 0.1, 1, 10, 100, 1000]}
grid_search_svm_d = GridSearchCV(svm_d, param_grid = param, cv = 5, n_jobs = -1)
grid_search_svm_d.fit(x_train_s, y_train)
svm_cv_param_d = grid_search_svm_d.cv_results_['params']
svm_cv_socre_mean_d = grid_search_svm_d.cv_results_['mean_test_score']
svm_cv_socre_std_d = grid_search_svm_d.cv_results_['std_test_score']
plt.plot(range(len(svm_cv_param_d)), svm_cv_socre_mean_d, color = 'blue')
plt.title('SVM')
plt.errorbar(range(len(svm_cv_param_d)), svm_cv_socre_mean_d, yerr = 2*svm_cv_socre_std_d, color = 'blue', ecolor='red')
plt.title('SVM')
print ("the best parameter is " , svm_cv_param_d[25])
svm_best_d = SVC(gamma = 0.01, C = 1000, probability = True).fit(x_train_s, y_train)
svm_score_d = svm_best_d.score(x_test_s, y_test)
svm_pre_d = svm_best_d.predict_proba(x_test_s)
#rf_pre = pd.DataFrame(rf_pre)
### Change 0 to not equal to 0 to caculate log loss
#rf_pre.loc[rf_pre[0] == 0, [0]] = 0.000000000001
## loss function value
z_svm_d = 0
for i in range(len(y_test)):
    z_svm_d = z_svm_d + y_test[i]*math.log(svm_pre_d[i][1]) + (1-y_test[i])*math.log(svm_pre_d[i][0])
log_loss_svm_d = -(z_svm_d/len(y_test))
print("The log loss for support vector machine with deep learning feature is ", log_loss_svm_d)
fin_result = pd.DataFrame(columns = ["MoldeType","Accuracy Without DeapLearning Feature","Accuracy With DeapLearning Feature", "Logloss Without DeapLearning Feature", "Logloss With DeapLearning Feature"])
fin_result.loc[0] = ["Logistic Regression", best_lor_acc, best_lor_d_acc, log_loss_lr, log_loss_lr_d]
fin_result.loc[1] = ["Random Forest", rf_score, rf_score_d, log_loss_rf, log_loss_rf_d]
fin_result.loc[2] = ["Support Vector Machine", svm_score, svm_score_d, log_loss_svm, log_loss_svm_d]
print(fin_result)


zzz = x_test['ARCII_deepModel_0']
result = []
for e in x_test['ARCII_deepModel_0']:
    if e > 0.5:
        result.append(1)
    else:
        result.append(0)
count = 0
count_acc = 0
for (e1, e2) in zip(result, y_test):
    if e1 == e2:
        count_acc += 1
    count += 1
print(count_acc/count)

z_dl = 0
zzz_list = zzz.tolist()
for i in range(len(zzz)):
    z_dl = z_dl + zzz_list[i]*math.log(zzz_list[i]) + (1-y_test[i])*math.log(1 - zzz_list[i])
log_loss_dl = -(z_dl/len(y_test))
print("The log loss for support vector machine with deep learning feature is ", log_loss_svm_d)