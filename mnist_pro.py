import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from torch import optim
import time
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
# sampler = torch.utils.data.SubsetRandomSampler(indices=list(range(100)))

writer = SummaryWriter('LeNet')
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 已给定的概率随即水平翻转给定的PIL图像
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize([0.406], [0.225])  # 用平均值和标准偏差归一化张量图像
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.406], [0.225])
    ]),
}
# download mnist dataset
data_train = torchvision.datasets.MNIST(root='./data/',train=True,download=True,transform=data_transforms['train'])
data_test = torchvision.datasets.MNIST(root='./data/',train=False,download=True,transform=data_transforms['test'])
datasets = {'train':data_train,'test':data_test}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=512,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'test']
               }
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test']}
class_names = datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):  # 正向传播过程
        x = self.block_1(x)
        x = x.view(-1,16*5*5)
        x = self.block_2(x)
        return x

def imshow(inp, title=None):
    # print(inp.size())
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 为了让图像更新可以暂停一会

"""
# 查看梯度图片
inputs, classes = next(iter(dataloaders['train'])) # 读取一个batch
out = torchvision.utils.make_grid(inputs) # 求梯度
imshow(out, title=[class_names[x] for x in classes])
"""
def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoc{}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        # epoch一次完成后切换到测试phase
        for phase in ['train','test']:
            if phase == 'train':
                model.train()   # 切换到train mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad() # 梯度清零
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    # backward + optimize only if in trainning phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
                lr = next(iter(optimizer.param_groups))['lr']
                print(lr)
                writer.add_scalar('Train/Loss',epoch_loss,epoch)
                writer.add_scalar('Train/Acc',epoch_acc,epoch)
            else:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
                writer.add_scalar('Test/Loss',epoch_loss,epoch)
                writer.add_scalar('Test/Acc',epoch_acc,epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_scheduler.step()  # 更新
        print()
    writer.close()
    time_elapsed = time.time() - since
    print('Trainning complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60 , time_elapsed % 60
    ))
    print('Best test Acc: {:4f}'.format(best_acc))

    return model
 
def visualize_model(model,num_images=6):
    was_trainning = model.training
    model.eval()

    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i,(inputs,labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _,preds = torch.max(outputs,1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predict: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode = was_trainning)
                    return
                model.train(mode = was_trainning)
if __name__ == '__main__':
    net = LeNet()
    net = net.to(device)
    lr_list = []
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.9)

    optimizer = optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9)
    net = train_model(net,criterion,optimizer,lr_scheduler,num_epochs=200)
    plt.plot(range(100),lr_list,color = 'r')




