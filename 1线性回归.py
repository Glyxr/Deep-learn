
import random
import torch
from d2l import torch as d2l
import numpy

#make datasets
def synthetic_data(w,b,num_examples):
    """"make noisy"""
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b;
    y += torch.normal(0,0.01,y.shape)
    # print(x)
    # print(y)
    # print(y.reshape((-1,1)))
    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)

print('features:',features[0],'\nlabel:',labels[0])

#read data
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # print(indices)
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size,num_examples)]
        )
        # print(batch_indices)
        yield features[batch_indices],labels[batch_indices]
    # batch_indices = torch.tensor(indices[0:min(0 + batch_size,num_examples)])
    # return features[batch_indices],labels[batch_indices]

batch_size = 10

# for x,y in data_iter(batch_size,features,labels):
#     print(x,'\n',y)
#     break
#init
w = torch.normal(0,0.01,size = (2,1),requires_grad= True)
b = torch.zeros(1,requires_grad= True)

#定义模型
def linreg(x,w,b):
    return torch.matmul(x,w) + b
#定义loss
def loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

#定义优化算法
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

#train
lr = 0.03
num_epochs = 3
net = linreg
loss = loss
for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,features,labels):
        l = loss(net(x,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss = {float(train_l.mean()):f}')
