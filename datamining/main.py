import Tovec
import deal_data
import trainModel

if __name__=='__main__':
    # Tovec.tovec_csv_udefsize(400,6,100)
    # deal_data.get_newdata()
    # vsize=100
    # train_arrays, train_labels, test_arrays, test_labels=trainModel.get_data('newdata.csv',vsize)
    # knn=trainModel.Knnclassifier(train_arrays, train_labels, test_arrays, test_labels)
    # trainModel.Grid_knn(vsize)

    # trainModel.get_pic_vsize()

    vsize=330
    Tovec.tovec_csv_udefsize(400,6,vsize)
    deal_data.get_newdata()
    train_arrays, train_labels, test_arrays, test_labels = trainModel.get_data('newdata.csv', vsize)
    knn = trainModel.Knnclassifier(train_arrays, train_labels, test_arrays, test_labels)
    svm=trainModel.SVM(train_arrays, train_labels, test_arrays, test_labels)

    #CV
    # vsize=330
    # train_arrays, train_labels, test_arrays, test_labels = trainModel.get_data('newdata.csv', vsize)
    # model= trainModel.SVCModel(train_arrays, train_labels, test_arrays, test_labels)
    # model2=trainModel.KnnModel(train_arrays, train_labels, test_arrays, test_labels)
    # s= trainModel.ModelCV(model, train_arrays, train_labels)
    # s2=trainModel.ModelCV(model2, train_arrays, train_labels)
    # print('means of SVM CV score：',s)
    # print('means of KNN CV score：', s2)