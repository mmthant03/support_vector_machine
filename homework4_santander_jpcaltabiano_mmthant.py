import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas

DEBUG = True

# Load data
d = pandas.read_csv('train.csv')
d = d.sample(frac=1)
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
# TODO
if DEBUG: print("Splitting into Train/Test folds.....")

train_y = y[:100000]
train_X = X[:100000]

test_y = y[100000:]
test_X = X[100000:]

train_subX = np.vsplit(train_X, 1000)
train_suby = np.split(train_y, 1000)

if DEBUG:
    print("Splitting into mini batches.....") 
    print("Size of a mini batch: ({},{})".format(len(train_subX), len(train_subX[0])))

# Linear SVM
# TODO
if DEBUG: print("Training Linear SVM model.....")
# Fitting and Training the model in linear SVC kernel
svms = []
for i in range(len(train_subX)):
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(train_subX[i], train_suby[i])
    svms.append(svm)

# Predicting the model
preds = np.zeros(test_y.shape)
count = 0
for i in svms[:5]:
    preds += i.decision_function(test_X)
    if DEBUG: 
        count += len(train_subX[0])
        print("Total Examples Being Trained: {}".format(count))

preds = preds/len(svms)
# preds = np.where(preds > 0.5, 1, 0)
# acc = np.mean(preds == test_y)
# if DEBUG: 
#     print(preds)
#     print(acc)

# Non-linear SVM (polynomial kernel)
# TODO
if DEBUG: print("Training Polynomial SVM model.....")
# Fitting and Training the model in linear SVC kernel
svms_poly = []
for i in range(len(train_subX)):
    svm = sklearn.svm.SVC(kernel='poly', C=1e15, gamma='scale')  # 1e15 -- approximate hard-margin
    svm.fit(train_subX[i], train_suby[i])
    svms_poly.append(svm)

# Predicting the model
preds_poly = np.zeros(test_y.shape)
count = 0
for i in svms_poly[:5]:
    preds_poly += i.decision_function(test_X)
    if DEBUG: 
        count += len(train_subX[0])
        print("Total Examples Being Trained: {}".format(count))

preds_poly = preds_poly/len(svms_poly)
# preds_poly = np.where(preds_poly > 0.5, 1, 0)
# acc_poly = np.mean(preds_poly == test_y)
# if DEBUG: 
#     print(preds_poly)
#     print(acc_poly)


# Apply the SVMs to the test set
yhat1 = preds  # Linear kernel
yhat2 = preds_poly  # Non-linear kernel

# Compute AUC
auc1 = sklearn.metrics.roc_auc_score(test_y, yhat1)
auc2 = sklearn.metrics.roc_auc_score(test_y, yhat2)

print(auc1)
print(auc2)
