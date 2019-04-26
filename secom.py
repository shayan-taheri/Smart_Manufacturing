import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.misc import imsave
from sklearn.model_selection import cross_val_score
import itertools
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import svm
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

data = np.genfromtxt('/home/ubuntu/PycharmProjects/MLProj/secom.txt', delimiter=' ')

dataset = read_csv('/home/ubuntu/PycharmProjects/MLProj/secom.txt', header=None, sep=' ')
null_sum = dataset.isnull().sum()

index = []
for i in range(len(null_sum)):
    if null_sum[i] >= 400:
        index.append(i)
# fig1 = plt.figure()
# for i in range(len(null_sum)):
#     plt.scatter(i, null_sum[i], marker='o', c='blue')


max = np.nanmax(data, axis=0)
min = np.nanmin(data, axis=0)

for i in range(len(null_sum)):
    if min[i] == max[i]:
        index.append(i)

index_unique = np.unique(index)

data_pruned = np.delete(data, index, axis=1)

null_sum_pruned = np.zeros(len(data_pruned[0]))
for i in range(len(data_pruned[0])):
    null_sum_pruned[i] = np.isnan(data_pruned[:, i]).sum()

mean = np.nanmean(data_pruned, axis=0)

data_filled = np.copy(data_pruned)
for feature_counter in range(len(data_pruned[0])):
    for instance_counter in range(len(data_pruned)):
        if np.isnan(data_pruned[instance_counter, feature_counter]):
            data_filled[instance_counter, feature_counter] = mean[feature_counter]


label = np.loadtxt('/home/ubuntu/PycharmProjects/MLProj/label.txt')

data_filled_norm = np.zeros(data_filled.shape)
maximum = np.nanmax(data_filled, axis=0)
minimum = np.nanmin(data_filled, axis=0)
for i in range(len(minimum)):
    data_filled_norm[:, i] = (data_filled[:, i] - minimum[i])/(maximum[i] - minimum[i]) * 2 - 1

# x_train, x_test, y_train, y_test = train_test_split(data_filled_norm, label, test_size=0.1, random_state=42)
# clf = RandomForestClassifier(random_state=0)
# clf.fit(x_train, y_train)
# y_pred = cross_val_predict(clf, x_test, y_test)
# print(confusion_matrix(y_test, y_pred))
#
# x_train, x_test, y_train, y_test = train_test_split(data_filled_norm, label, test_size=0.1, random_state=42)
# clf = svm.SVC(kernel='linear')
# clf.fit(x_train, y_train)
# y_pred = cross_val_predict(clf, x_test, y_test)
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
#
# data_filled_shuffled, label_shuffled = shuffle(data_filled_norm, label, random_state=0)
# # clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = svm.SVC(kernel='linear', C=1000000)
# y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
# print(accuracy_score(label_shuffled, y_pred))
# print(confusion_matrix(label_shuffled, y_pred))
#
data_filled_shuffled, label_shuffled = shuffle(data_filled_norm, label, random_state=0)
clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print(accuracy_score(label_shuffled, y_pred))
print(confusion_matrix(label_shuffled, y_pred))

data_pruned_norm = np.zeros(data_pruned.shape)
maximum = np.nanmax(data_pruned, axis=0)
minimum = np.nanmin(data_pruned, axis=0)
for i in range(len(minimum)):
    data_pruned_norm[:, i] = (data_pruned[:, i] - minimum[i])/(maximum[i] - minimum[i]) * 2 - 1

iteration = int(np.sum(null_sum_pruned))

confidence = 10000 * np.ones(data_pruned.shape)
sum_row = np.zeros(len(data_pruned))
sum_column = np.zeros(len(data_pruned[0]))
for i in range(len(data_pruned)):
    sum_row[i] = np.isnan(data_pruned[i, :]).sum()
for j in range(len(data_pruned[0])):
    sum_column[j] = np.isnan(data_pruned[:, j]).sum()

for i in range(len(data_pruned)):
    for j in range(len(data_pruned[0])):
        if np.isnan(data_pruned[i, j]):
            confidence[i, j] = sum_row[i] + sum_column[j]

for counter in range(iteration):
    target = np.argmin(confidence)
    x = int(target / len(data_pruned[0]))
    y = int(target % len(data_pruned[0]))

    label_target = data_pruned_norm[:, y]
    data_target = np.zeros((len(data_pruned), len(data_pruned[0]) + 1))
    data_target[:, 0:len(data_pruned[0])] = data_pruned_norm
    data_target[:, len(data_pruned[0])] = label
    data_target = np.delete(data_target, y, axis=1)
    x_test = data_target[x, :]
    y_test = label_target[x]
    x_train = np.delete(data_target, x, axis=0)
    y_train = np.delete(label_target, x)

    index = []
    for j in range(len(y_train)):
        if np.isnan(y_train[j]):
            index.append(j)
    x_train = np.delete(x_train, index, axis=0)
    y_train = np.delete(y_train, index)

    mean_train = np.nanmean(x_train, axis=0)
    for feature_counter in range(len(x_train[0])):
        for instance_counter in range(len(x_train)):
            if np.isnan(x_train[instance_counter, feature_counter]):
                x_train[instance_counter, feature_counter] = mean_train[feature_counter]
    for feature_counter in range(len(x_test)):
        if np.isnan(x_test[feature_counter]):
            x_test[feature_counter] = mean_train[feature_counter]

    clf = KNeighborsRegressor(n_neighbors=3)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test.reshape(1, -1))
    for j in range(len(data_pruned[0])):
        if np.isnan(data_pruned[x, j]):
            confidence[x, j] -= 1
    for i in range(len(data_pruned)):
        if np.isnan(data_pruned[i, y]):
            confidence[i, y] -= 1
    confidence[x, y] = 10000
    data_pruned_norm[x, y] = prediction
    print('data_pruned_norm['+str(x)+', ' + str(y) +'] = ' + str(prediction))
    if counter % 500 == 0:
        print(counter)

null_sum_filled = np.zeros(len(data_pruned_norm[0]))
for i in range(len(data_pruned_norm[0])):
    null_sum_filled[i] = np.isnan(data_pruned_norm[:, i]).sum()

data_pruned_shuffled, label_shuffled = shuffle(data_pruned_norm, label, random_state=0)
clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, data_pruned_shuffled, label_shuffled, cv=10)
print(accuracy_score(label_shuffled, y_pred))
print(confusion_matrix(label_shuffled, y_pred))




file = open('/home/ubuntu/PycharmProjects/MLProj/knn_filled', 'w+')
np.savetxt(file, data_pruned_norm, delimiter=',')
file.close()




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label', labelpad=1)


class_names = ['Pass', 'Fail']

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=False,
                      title='"Mean" Method Confusion Matrix')
plt.savefig('/home/ubuntu/PycharmProjects/MLProj/confusion_base.png', format='png')
plt.show()

cnf_matrix = confusion_matrix(label_shuffled, y_pred)
class_names = ['Pass', 'Fail']

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=False,
                      title='"My KNN Algorithm" Method with SMOTE and RF Confusion Matrix')
plt.savefig('/home/ubuntu/PycharmProjects/MLProj/confusion_new.png', format='png')
plt.show()


file = open('/home/ubuntu/PycharmProjects/MLProj/knn_filled', 'r')
data_pruned_norm = np.loadtxt(file, delimiter=',')
file.close()

data_pruned_shuffled, label_shuffled = shuffle(data_pruned_norm, label, random_state=0)
lsvc = linear_model.LogisticRegression(penalty="l2", dual=False).fit(data_pruned_shuffled, label_shuffled)
model = SelectFromModel(lsvc, prefit=True)
data_pruned_new = model.transform(data_pruned_shuffled)

lsvc = linear_model.LogisticRegression(penalty="l2", dual=False).fit(data_filled_shuffled, label_shuffled)
model = SelectFromModel(lsvc, prefit=True)
data_filled_new = model.transform(data_filled_shuffled)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(data_filled_new, label_shuffled)

#######
clf = linear_model.LogisticRegression()
data_filled_shuffled, label_shuffled = shuffle(data_filled_norm, label, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by logistic regression:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))

clf = RandomForestClassifier()
data_filled_shuffled, label_shuffled = shuffle(data_filled_norm, label, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by RF:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))


clf = LinearSVC()
data_filled_shuffled, label_shuffled = shuffle(data_filled_norm, label, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by SVM:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))

clf = KNeighborsClassifier(n_neighbors=3)
data_filled_shuffled, label_shuffled = shuffle(data_filled_norm, label, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by KNN:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))
#######

clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, data_filled_new, label_shuffled, cv=10)
print('Filled with mean and selected by logistic regression and classified by logistic regression:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))

clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, data_filled_new, label_shuffled, cv=10)
print('Filled with mean and selected by logistic regression and classified by RF:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))


clf = LinearSVC()
y_pred = cross_val_predict(clf, data_filled_new, label_shuffled, cv=10)
print('Filled with mean and selected by logistic regression and classified by SVM:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))


clf = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(clf, data_filled_new, label_shuffled, cv=10)
print('Filled with mean and selected by logistic regression and classified by KNN:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))
######


clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, data_pruned_shuffled, label_shuffled, cv=10)
print('Filled with KNN and classified by logistic regression:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))


clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, data_pruned_shuffled, label_shuffled, cv=10)
print('Filled with KNN and classified by RF:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))


clf = LinearSVC()
y_pred = cross_val_predict(clf, data_pruned_shuffled, label_shuffled, cv=10)
print('Filled with KNN and classified by SVM:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))


clf = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(clf, data_pruned_shuffled, label_shuffled, cv=10)
print('Filled with KNN and classified by KNN:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, accuracy_score(label_shuffled, y_pred), precision_score(label_shuffled, y_pred), recall_score(label_shuffled, y_pred), f1_score(label_shuffled, y_pred))

#######
clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, data_pruned_new, label_shuffled, cv=10)
print('Filled with KNN and selected by logistic regression and classified by logistic regression:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))

clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, data_pruned_new, label_shuffled, cv=10)
print('Filled with KNN and selected by logistic regression and classified by RF:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))

clf = LinearSVC()
y_pred = cross_val_predict(clf, data_pruned_new, label_shuffled, cv=10)
print('Filled with KNN and selected by logistic regression and classified by SVC:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))

clf = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(clf, data_pruned_new, label_shuffled, cv=10)
print('Filled with KNN and selected by logistic regression and classified by KNN:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))

######

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(data_filled_norm, label)

clf = linear_model.LogisticRegression()
data_filled_shuffled, label_shuffled = shuffle(X_res, y_res, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by logistic regression:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))

clf = RandomForestClassifier()
data_filled_shuffled, label_shuffled = shuffle(X_res, y_res, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by RF:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))


clf = LinearSVC()
data_filled_shuffled, label_shuffled = shuffle(X_res, y_res, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by SVM:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))

clf = KNeighborsClassifier(n_neighbors=3)
data_filled_shuffled, label_shuffled = shuffle(X_res, y_res, random_state=0)
y_pred = cross_val_predict(clf, data_filled_shuffled, label_shuffled, cv=10)
print('Filled with mean and classified by KNN:')
cm = confusion_matrix(label_shuffled, y_pred)
print(confusion_matrix(label_shuffled, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(label_shuffled, y_pred), 100 * precision_score(label_shuffled, y_pred), 100 * recall_score(label_shuffled, y_pred), 100 * f1_score(label_shuffled, y_pred))
#######

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(data_filled_new, label_shuffled)

clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with mean and selected by logistic regression and classified by logistic regression:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with mean and selected by logistic regression and classified by RF:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))


clf = LinearSVC()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with mean and selected by logistic regression and classified by SVM:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))


clf = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with mean and selected by logistic regression and classified by KNN:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

######

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(data_pruned_shuffled, label_shuffled)


clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and classified by logistic regression:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))


clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and classified by RF:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))


clf = LinearSVC()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and classified by SVM:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))


clf = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and classified by KNN:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

######

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(data_pruned_new, label_shuffled)

clf = linear_model.LogisticRegression()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and selected by logistic regression and classified by logistic regression:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

clf = RandomForestClassifier()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and selected by logistic regression and classified by RF:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

clf = LinearSVC()
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and selected by logistic regression and classified by SVC:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

clf = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(clf, X_res, y_res, cv=10)
print('Filled with KNN and selected by logistic regression and classified by KNN:')
cm = confusion_matrix(y_res, y_pred)
print(confusion_matrix(y_res, y_pred))
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]
print(TP, TN, FP, FN, TP+TN+FP+FN, 100 * accuracy_score(y_res, y_pred), 100 * precision_score(y_res, y_pred), 100 * recall_score(y_res, y_pred), 100 * f1_score(y_res, y_pred))

######