import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import Tovec
import deal_data

import matplotlib.pyplot as plt

def get_data(path,vsize):
    data=pd.read_csv(path)
    npdata=data.values
    vsize=vsize+1
    X_data=npdata[:,1:vsize]
    Y_data=npdata[:,vsize]
    X_data=preprocessing.scale(X_data)

    train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_data, Y_data, test_size=0.25,
                                                                            random_state=6)
    return train_arrays, train_labels,  test_arrays,  test_labels #训练数据 训练标签，测试数据，测试标签

def get_data_vec(dfdata,vsize):
    npdata=dfdata.values
    vsize=vsize+1
    X_data = npdata[:, 1:vsize]
    Y_data = npdata[:, vsize]
    X_data = preprocessing.scale(X_data)
    train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_data, Y_data, test_size=0.25,
                                                                            random_state=6)
    return train_arrays, train_labels, test_arrays, test_labels  # 训练数据 训练标签，测试数据，测试标签

def SVM(train_arrays, train_labels,  test_arrays,  test_labels):
    classifier = SVC(C=2, gamma=0.0001, kernel='rbf', class_weight='balanced')
    classifier.fit(train_arrays, train_labels.astype(int))
    y = classifier.predict(test_arrays)

    print('SVM_accuracy', metrics.accuracy_score(test_labels.astype(int), y))
    print('SVM_F1scores',metrics.f1_score(test_labels.astype(int),y,average='macro'))
    # print('Kappa', metrics.cohen_kappa_score(test_labels.astype(int), y))
    return metrics.accuracy_score(test_labels.astype(int), y)

def SVCModel(train_arrays, train_labels,  test_arrays,  test_labels):
    classifier = SVC(C=2, gamma=0.0001, kernel='rbf', class_weight='balanced')
    classifier.fit(train_arrays, train_labels.astype(int))
    return classifier

def RFclassifier(train_arrays, train_labels, test_arrays, test_label):
    classifier = RandomForestClassifier(oob_score=True, random_state=10, max_depth=7, n_estimators=30)
    classifier.fit(train_arrays, train_labels.astype(int))
    print('oob', classifier.oob_score_)
    y = classifier.predict(test_arrays)
    print('RFaccuracy', metrics.accuracy_score(test_label.astype(int), y))
    return metrics.accuracy_score(test_label.astype(int), y)

def Knnclassifier(train_arrays, train_labels, test_arrays, test_label):
    classifier = KNeighborsClassifier(n_neighbors=21, algorithm='auto',weights='uniform')
    classifier.fit(train_arrays, train_labels.astype(int))
    y = classifier.predict(test_arrays)
    print('KNN_accuracy', metrics.accuracy_score(test_label.astype(int), y))
    print('KNN_F1scores', metrics.f1_score(test_label.astype(int), y, average='macro'))
    return metrics.accuracy_score(test_label.astype(int), y)

def KnnModel(train_arrays, train_labels, test_arrays, test_label):
    classifier = KNeighborsClassifier(n_neighbors=21, algorithm='auto',weights='uniform')
    classifier.fit(train_arrays, train_labels.astype(int))
    return classifier

def ModelCV(self,model):
    X,y=self.get_data()
    X=preprocessing.scale(X)
    train_arrays, test_arrays, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=6)
    kfold = KFold(n_splits=10, shuffle=True, random_state=37)
    scores2 = cross_val_score(model, train_arrays, train_labels, cv=kfold)
    scores2=np.mean(scores2)
    return scores2


def Grid(vsize):

    train_arrays, train_labels, test_arrays, test_labels=get_data('newdata.csv',vsize)

    C = [1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]
    gamma = [0.001, 0.0001]
    param_test = dict(C=C, gamma=gamma)
    clf = SVC(kernel='rbf', class_weight='balanced')
    gridSVC = GridSearchCV(clf, param_grid=param_test, cv=5, scoring='accuracy')
    gridSVC.fit(train_arrays, train_labels.astype(int))
    print('best score is:', str(gridSVC.best_score_))
    print('best params are', str(gridSVC.best_params_))

def Grid_knn(vsize):

    train_arrays, train_labels, test_arrays, test_labels=get_data('newdata.csv',vsize)

    n_neighbors=list(range(1,101,5))
    weights=['distance','uniform']
    algorithm = ['auto','ball_tree', 'kd_tree', 'brute']

    param_test = dict(n_neighbors=n_neighbors, weights=weights,algorithm=algorithm)
    clf = KNeighborsClassifier()
    gridSVC = GridSearchCV(clf, param_grid=param_test, cv=5, scoring='accuracy')
    gridSVC.fit(train_arrays, train_labels.astype(int))
    print('best score is:', str(gridSVC.best_score_))
    print('best params are', str(gridSVC.best_params_))


def get_pic_vsize():

    knn_list=[]
    svm_list=[]
    x_axis=[]
    for vsize in range(300,501,10):
        data = Tovec.tovec_extract(400, 6,vsize)
        dfdata = deal_data.get_newdata_byvec(data)
        train_arrays, train_labels, test_arrays, test_labels = get_data_vec(dfdata,vsize)
        knn = Knnclassifier(train_arrays, train_labels, test_arrays, test_labels)
        svm=SVM(train_arrays, train_labels, test_arrays, test_labels)
        knn_list.append(knn)
        svm_list.append(svm)
        x_axis.append(vsize)
    plt.figure()
    plt.plot(x_axis, svm_list, color='green', label='svm')
    plt.plot(x_axis, knn_list, color='red', label='knn')
    plt.legend()
    plt.show()

def ModelCV(model,train_arrays, train_labels):
    kfold = KFold(n_splits=10, shuffle=True, random_state=37)
    scores2 = cross_val_score(model, train_arrays, train_labels.astype(int), cv=kfold)
    scores2=np.mean(scores2)
    return scores2


