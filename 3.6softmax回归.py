import matplotlib.pyplot as plt
import torch
from IPython import display
from d2l import torch as d2l


batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = torch.normal(0,0.01,size = (num_inputs,num_outputs),requires_grad = True)
b = torch.zeros(num_outputs,requires_grad = True)

x = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
x.sum(0,keepdim = True),x.sum(1,keepdim = True) #0-【5，7，9】  1-【6，15】

def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1,keepdim = True)
    return x_exp / partition

x = torch.normal(0,1,(2,5))
x_prob = softmax(x)
x_prob,x_prob.sum(1)

#define model
def net(x):
    return softmax(torch.matmul(x.reshape((-1,w.shape[0])), w) + b)

#交叉熵损失
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

#计算精确度
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] >1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    """计算指定数据集上模型的精度"""
    if isinstance(net,torch.nn.Module):
        net.val()
    metric = Accumulator(2)
    with torch.no_grad():
        for x,y in data_iter:
            metric.add(accuracy(net(x),y),y.numel())

    return metric[0]/metric[1]


class Accumulator:
    def __init__(self,n):
        self.data = [0.0] * n
    def add(self,*args):
       self.data = [a + float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

#
def train_epoch_ch3(net,train_iter,loss,updater): #@save
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for x,y in train_iter:
        y_hat = net(x)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        else:
            #使用定制的优化器和损失函数
            l.sum().backward()
            updater(x.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    #返回训练损失和训练精度
    return metric[0]/metric[2],metric[1]/metric[2]

# class Animator:
#     def __init__(self,xlabel = None,ylabel = None,legend = None,xlim = None,
#                  ylim = None,xscale = 'linear',yscale = 'linear',fimt = ('-','m--','g-.','r:'),
#                  nrows = 1,ncols = 1,figsize = (3.5,2.5)):
#         if legend is None:
#             legend = []
#         d2l.use_svg_display() # use_svg_display函数指定matplotlib软件包输出svg图表以获得更清晰的图像
#         self.fig,self.axes = d2l.plt.subplots(nrows,ncols,figsize = figsize)
#         if nrows * ncols == 1:
#             self.axes = [self.axes,]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    plt.show()
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()

predict_ch3(net, test_iter)
