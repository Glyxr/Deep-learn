import hashlib
import os
import tarfile
import zipfile
import requests



# 下载和缓存数据集
#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name,cache_dir = os.path.join('..','data')):  #@save
    '''下载一个DATA_HUB中的文件，返回本地文件名'''
    assert name in DATA_HUB,f"{name} 不存在于 {DATA_HUB}"
    url,shal_hash = DATA_HUB[name]   #数据集和密钥
    os.makedirs(cache_dir,exist_ok=True)
    fname = os.path.join(cache_dir,url.split('/')[-1])
    if os.path.exists(fname):
        shal = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                shal.update(data)
            if shal.hexdigest() == shal_hash:   #匹配密钥
                return fname #命中缓存
        print(f'正在从{url}下载{fname}...')
        r = requests.get(url,stream=True,verify=True)
        with open(fname,'wb') as f:
            f.write(r.content)
        return fname


def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


#访问和读取数据集
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l        #为什么要这样引用
#
import matplotlib.pyplot as plt
DATA_HUB['kaggle_house_train'] = ( #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (#@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

#利用pandas加载csv文件
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# print(train_data.shape)   #(1460,81)
# print(test_data.shape)    #(1459,80)
# print(torch.tensor([1,2]).shape)

all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
#std()求标准差
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]   #训练集的个数-1460
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32) #训练集
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)  #测试集
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)   #训练集的结果

#训练
loss = nn.MSELoss()
in_features = train_features.shape[1] #代表331个特征

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features,160),
        nn.ReLU(),
        nn.Linear(160,40),
        nn.ReLU(),
        nn.Linear(40,1)
    )
    return net

def log_rmse(net,features,labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                            torch.log(labels)))
    return rmse.item()


def train(net,train_features,train_labels,test_features,test_labels,
          num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[] #存放loss
    train_iter = d2l.load_array((train_features,train_labels),batch_size) #训练集
    #使用adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay) #优化器
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_features is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))

    return train_ls,test_ls


def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0] // k  #分成K份
    X_train,y_train = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part = X[idx,:],y[idx]
        if j == i:
            X_valid,y_valid = X_part,y_part    #验证集
        elif X_train is None:
            X_train,y_train = X_part,y_part   #训练集
        else:
            X_train = torch.cat([X_train,X_part],0)
            y_train = torch.cat([y_train,y_part],0)
    return X_train,y_train,X_valid,y_valid


def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum = 0,0   #训练,验证总损失
    for i in range(k):
        data = get_k_fold_data(k,i,X_train,y_train)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size) #训练,验证每一轮的损失
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1,num_epochs+1)),[train_ls,valid_ls],xlabel = 'epoch',ylabel = 'rmse',xlim = [1,num_epochs]
                     ,legend = ['train','valid'],yscale = 'log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k,num_epochs,lr,weight_decay,batch_size = 5,100,0.5,0,64
train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
# plt.show()

def train_and_pred(train_features,tset_features,train_labels
                   ,test_data,num_epochs,lr,weight_decay,batch_szie):
    net = get_net()
    train_ls,_ = train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.plot(np.arange(1,num_epochs+1),[train_ls],xlabel = 'epoch',ylabel = 'log rmse',xlim = [1,num_epochs],yscale='log')
    print(f'训练log rmse:{float(train_ls[-1]):f}')
    #将网络运用于测试集
    preds = net(test_features).detach().numpy()
    #将其格式化以导出到kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis = 1)
    submission.to_csv('submission.csv',index = False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
plt.show()






