#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


train_data = pd.read_csv('/home/zeriwang/2024Fall/2024Fall_UCAS_ML/2024Fall_UCAS_Introduction_to_Machine_Learning/2022CCF-BDCI-Home-development-crowd-forecast/dataTrain.csv')
test_data = pd.read_csv('/home/zeriwang/2024Fall/2024Fall_UCAS_ML/2024Fall_UCAS_Introduction_to_Machine_Learning/2022CCF-BDCI-Home-development-crowd-forecast/dataB.csv')
submission = pd.read_csv('/home/zeriwang/2024Fall/2024Fall_UCAS_ML/2024Fall_UCAS_Introduction_to_Machine_Learning/2022CCF-BDCI-Home-development-crowd-forecast/submit_example_B.csv')
data_nolabel = pd.read_csv('/home/zeriwang/2024Fall/2024Fall_UCAS_ML/2024Fall_UCAS_Introduction_to_Machine_Learning/2022CCF-BDCI-Home-development-crowd-forecast/dataNoLabel.csv')


print(f'train_data.shape = {train_data.shape}\ntest_data.shape  = {test_data.shape}')


train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']

loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)

com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)


cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])


num_columns = [ col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]


train = train[:50000]
label = label[:50000]


def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold, shuffle=True)

    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train,y_train)

        y_pred = model.predict_proba(x_test)[:,1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test,y_pred)
        print("Model = %s, KFold = %d, val_auc = %.4f" % (model_name, k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold


estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=200, random_state=42)),
        ('nb', GaussianNB()),
        ('lda', LinearDiscriminantAnalysis()),
        ('ridge', RidgeClassifier())
    ]
clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression()
)


X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)


clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)


features = []
feature_importances = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        features.append(col)
    feature_importances.append([col, auc1, auc1 - auc])


feature_importances.sort(key=lambda x: x[2])
for fi in feature_importances:
    print("| %10s | %.8f | %.8f |" % (fi[0], fi[1], fi[2]))


clf.fit(X_train[features], y_train)
y_pred = clf.predict_proba(X_test[features])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)


train = train[features]
test = test[features]
preds = model_train(clf, "StackingClassifier", 10)


submission['label'] = preds


submission.to_csv('submission_test.csv', index=False)