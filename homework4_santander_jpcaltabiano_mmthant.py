import sklearn.svm
import sklearn.metrics
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas

# Load data
d = pandas.read_csv('train.csv')
d = d.sample(frac=1)
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
train_y = y[:100000]
train_X = X[:100000]

test_y = y[100000:]
test_X = X[100000:]

train_subX = np.vsplit(train_X, 20)
train_suby = np.split(train_y, 20)

# Linear SVM
svms = []
for i in range(len(train_subX)):
    svm = sklearn.svm.LinearSVC(C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(train_subX[i], train_suby[i])
    svms.append(svm)

# Predicting the model
preds = np.zeros(test_y.shape)
for i in svms[:5]:
    preds += i.decision_function(test_X)
preds = preds/len(svms)

# Non-linear SVM (polynomial kernel)
svms_poly = []
for i in range(len(train_subX)):
    svm = sklearn.svm.SVC(kernel='poly', C=1e15, gamma='scale', degree=3)  # 1e15 -- approximate hard-margin
    svm.fit(train_subX[i], train_suby[i])
    svms_poly.append(svm)

# Predicting the model
preds_poly = np.zeros(test_y.shape)
for i in svms_poly[:5]:
    preds_poly += i.decision_function(test_X)
preds_poly = preds_poly/len(svms_poly)

# Apply the SVMs to the test set
yhat1 = preds  # Linear kernel
yhat2 = preds_poly  # Non-linear kernel

# Compute AUC
auc1 = sklearn.metrics.roc_auc_score(test_y, yhat1)
auc2 = sklearn.metrics.roc_auc_score(test_y, yhat2)

print(auc1)
print(auc2)