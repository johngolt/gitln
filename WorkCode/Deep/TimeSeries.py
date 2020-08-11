import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


'''
1，准备数据
2，定义模型
3，训练模型
4，评估模型
5，使用模型
6，保存模型。
'''

#准备数据
class CovidDataset(Dataset):
    '''torch.utils.data.Dataset是一个抽象类，用户想要加载自定义的数据只需要继承这个类，并且覆写其中的两个方法即可：
__len__:实现len(dataset)返回整个数据集的大小。
__getitem__:用来获取一些索引的数据，使dataset[i]返回数据集中第i个样本。'''

    def __init__(self, data, windows=8):
        self.data = data
        self.windows = windows

    def __len__(self):
        return len(self.data) - self.windows

    def __getitem__(self, i):
        x = self.data.loc[i:i+self.windows-1, :]
        feature = torch.tensor(x.to_numpy())
        y = self.data.loc[i+self.windows, :]
        label = torch.tensor(y.to_numpy())
        return (feature, label)


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, x, x_input):
        x_out = torch.max((1+x)*x_input[:,-1,:],torch.tensor(0.))
        return x_out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=5, batch_first=True)
        self.linear = nn.Linear(3,3)
        self.block = Block()
    
    def forward(self, x_input):
        x = self.lstm(x_input)[:,-1,:]
        x = self.linear(x)
        y = self.block(x, x_input)
        return y


net = Net()
model = torchkeras.Model(net)
print(model)

model.summary(input_shape=(8,3),input_dtype = torch.FloatTensor)
def mspe(y_pred,y_true):
    err_percent = (y_true - y_pred)**2/(torch.max(y_true**2,torch.tensor(1e-7)))
    return torch.mean(err_percent)

model.compile(loss_func = mspe,optimizer = torch.optim.Adagrad(model.parameters(),lr = 0.1))
dfhistory = model.fit(100,dl_train,log_step_freq=10)

# 保存模型参数

torch.save(model.net.state_dict(), "./data/model_parameter.pkl")

net_clone = Net()
net_clone.load_state_dict(torch.load("./data/model_parameter.pkl"))
model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func = mspe)

# 评估模型
model_clone.evaluate(dl_train)