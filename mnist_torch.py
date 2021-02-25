import os
import torch
import torchvision
from torchsummary import summary
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as dataloader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

# hyper parameters
path_model = "./checkpoint/"
batch_size = 512
epochs = 50
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# sampler = torch.utils.data.SubsetRandomSampler(indices=list(range(2000)))

# download mnist dataset
data_train = torchvision.datasets.MNIST(root='./data/',train=True,download=True,transform=transform)
data_test = torchvision.datasets.MNIST(root='./data/',train=False,download=True,transform=transform)

class_names = data_train.classes  # 获取数据集的分类信息 返回一个字典
# load dataset
data_train = dataloader(dataset=data_train,batch_size=batch_size,shuffle=False)
data_test = dataloader(dataset=data_test,batch_size=batch_size,shuffle=True)

"""
# 查看测试数据的构成
examples = enumerate(data_test)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
"""
def test():
    net.eval()  # 切换到测试模式
    test_correct_num = 0
    with torch.no_grad():   # 不更新参数

        for batch_idx,(data,target) in enumerate(data_test):
            # data = data.to(device)
            # target = target.to(device)
            output = net(data) # 正向传播得到预测值
            _, pred = torch.max(output.data, 1)
            test_correct_num += torch.sum(pred==target).item()
            print("Test Epoch:{} [{}/{} ({:.0f}%)]\t acc:{:.2f}".format(epoch,batch_idx*batch_size,len(data_test.dataset),
                                                 100. * batch_size*batch_idx/len(data_test.dataset),test_correct_num/len(data_test.dataset)))
def train():

    for batch_idx, (data, target) in enumerate(data_train):
        # 清除grad累积值
        optimizer.zero_grad()
        # 读取dataloader中的数据，前半部分是tensor变量，后半部分是真实label
        data = data.to(device)
        target = target.to(device)
        # forward之后得到预测值
        output = net(data)
        # 计算loss
        loss = cost_fun(output, target)
        # backward
        loss.backward()
        # 收集一组新的梯度，并使用optimizer.step()将其传播回每个网络参数
        optimizer.step()
        # 给出loss和acc
        train_loss.append(loss)
        _, pred = torch.max(output.data, 1)
        correct_num = torch.sum(pred == target).item()
        train_acc.append(correct_num / batch_size)
        print("Train Epoch:{}[{}/{} ({:.0f}%)]\t Loss:{:.6f} acc:{:.2f}".format(epoch, batch_idx * batch_size,
               len(data_train.dataset),100. * batch_size * batch_idx / len(data_train.dataset), loss.item(),correct_num / batch_size))

def save_state():
    print('===> Saving weights...')
    state = {
        'state': net.state_dict(),
        'epoch': epoch  # 将epoch一并保存
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, path_model + 'Epoch:' + str(epoch) + ' Loss:' + str(train_loss[-1].item()) + '.pth')

def predict():
    state_path = './checkpoint/***' #  ***为指定加载的权重文件名称
    print('===> Loading weights : ' + state_path)
    torch.load(state_path)  # 加载最后训练出的权重
    # 从测试集中选取一个batch做预测
    pred_test = enumerate(data_test)
    batch_idx, (pred_data, pred_gt) = next(pred_test)
    output = net(pred_data)
    _, pred = torch.max(output.data, 1) # 得到预测值
    print("ground truth: ",pred_gt)
    print("predict value: ",pred)

# 构建网络
class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):						# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 输出为6*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 输出为16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为16*5*5
        )
        self.block_2 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):  # 正向传播过程
        x = self.block_1(x)
        x = x.view(-1,16*5*5)
        x = self.block_2(x)
        return

# 显示各层参数量和训练过程batch所占内存大小
def variaes_show():
    net = LeNet()
    data_input = Variable(torch.randn(16,1,28,28))
    print(data_input.size())
    net(data_input)
    print(summary(net,(1,28,28)))
if __name__ == '__main__':
    writer = SummaryWriter('logs')
    device = torch.device('cuda:0')
    net = LeNet()  # 实例化网络
    # data_input = Variable(torch.randn(16,1,28,28))
    # print(net(data_input))
    net.to(device) # 将参数送入GPU中
    cost_fun = nn.CrossEntropyLoss()
    # optim
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.95, weight_decay=1e-3)

    # train
    for epoch in range(epochs):

        # train
        train_loss = []
        train_acc = []
        # train
        train()
        writer.add_scalar('Train/Loss', train_loss[-2].item(), epoch)
        writer.add_scalar('Train/Acc', train_acc[-2], epoch)

        # ----------------------------------------- #
        # save_state
        # ----------------------------------------- #
        print('===> Saving models...')
        state = {
            'state': net.state_dict(),
            'epoch': epoch  # 将epoch一并保存
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('./checkpoint')
        torch.save(state, path_model + 'Epoch:' + str(epoch) + ' Loss:'+ str(train_loss[-1].item()) + '.pth')

        # ----------------------------------------- #
        # test
        # ----------------------------------------- #
        test()
    writer.close()

    # ----------------------------------------- #
    # 加载指定的weights进行预测
    # ----------------------------------------- #
    # predict()
