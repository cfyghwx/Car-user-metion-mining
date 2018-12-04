import pandas as pd
import numpy as np

def get_newdata(path='new.csv'):
    data1=pd.read_csv('new.csv',header=None)
    npdata1=data1.values
    f=open('D:\\南京大学\\研一\\上课材料+作业\\作业\\data mining\\大作业\\train.csv',encoding='utf-8')
    #data为pandas对象，类似于表格的样子
    data=pd.read_csv(f)
    npdata=data_replace_topic(data)
    print(npdata)

    uid=npdata[:,0]
    newdata=np.c_[uid,npdata1]
    newdata=np.c_[newdata,npdata[:,2:]]
    # print(newdata.shape)
    df=pd.DataFrame(newdata)
    print(df)
    # np.savetxt('newdata.csv', newdata, delimiter = ',')
    df.to_csv('newdata.csv',encoding= 'utf-8',index=False)

def get_newdata_byvec(vecdata):
    npdata1=vecdata
    f=open('D:\\南京大学\\研一\\上课材料+作业\\作业\\data mining\\大作业\\train.csv',encoding='utf-8')
    #data为pandas对象，类似于表格的样子
    data=pd.read_csv(f)
    npdata=data_replace_topic(data)
    # print(npdata)

    uid=npdata[:,0]
    newdata=np.c_[uid,npdata1]
    newdata=np.c_[newdata,npdata[:,2:]]
    # print(newdata.shape)
    df=pd.DataFrame(newdata)
    return df

def data_replace_topic(data):
    subjectlist=['动力','价格','内饰','配置','安全性','外观','操控','油耗','空间','舒适性']
    for i in range(len(subjectlist)):
        data.ix[data['subject']==subjectlist[i],'subject']=i
        # print(i,'----',subjectlist[i])

    npdata=data.values
    return npdata
