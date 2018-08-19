import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transform
from SENet import *
import visdom
import numpy as np
from torchnet import meter
import sys


if torch.cuda.is_available():
    device = 'cuda'
    print("you have a gpu")
else:
    device = 'cpu'
    print("you should buy a gpu")
transform_train = transform.Compose([
    transform.RandomCrop(32, padding=4),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize((0.49139968, 0.48215841, 0.44653091),
                        (0.24703223, 0.24348513, 0.26158784))
])
transform_test = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.49139968, 0.48215841, 0.44653091),
                        (0.24703223, 0.24348513, 0.26158784))
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)
#net = SENet18()
net = SENet34()
best_net = None
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005,
                      momentum=0.9, weight_decay=5e-4)
vis = visdom.Visdom(env='cifar')
train_loss_meter = meter.AverageValueMeter()
train_acc_meter = meter.AverageValueMeter()
best_acc = 0
pre_acc = 0
cur_acc = 0
dec_num = 0


def train(epoch):
    global train_loss_meter, train_acc_meter
    train_loss_meter.reset()
    train_acc_meter.reset()
    print('Epoch:%d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alltotal = 0
    for batch_idx, (inputs, targets)in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_loss_meter.add(train_loss)
        _, predicted = outputs.max(1)
        alltotal += targets.size(0)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        train_acc_meter.add(correct/total)
        print('(%d/%d)Loss:%.3f | Acc: %.3f%%' % (alltotal/targets.size(0), 50000.0 /
                                                  targets.size(0), train_loss, 100*correct/total))
    return train_loss_meter.value()[0], train_acc_meter.value()[0]


def test(epoch):
    global best_acc, dec_num, cur_acc, pre_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    alltotal = 0
    step = 0
    acc_all = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets)in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total = targets.size(0)
            step += 1
            alltotal += targets.size(0)
            correct = predicted.eq(targets).sum().item()
            acc = correct/targets.size(0)
            acc_all += acc
            print('(%d/%d)Loss:%.3f | Acc:%.3f%%' % (alltotal/targets.size(0), 10000.0 /
                                                     targets.size(0), test_loss/step, 100*acc_all/step))
    pre_acc = cur_acc
    cur_acc = acc_all/step
    if acc_all/step > best_acc:
        best_acc = acc_all/step
        dec_num = 0
        best_net = net
    if pre_acc > cur_acc:
        dec_num += 1
    if dec_num == 3:
        torch.save(best_net.state_dict(), 'SENet_Model.pt')
        print("save finished!")
        sys.exit(0)
    return test_loss/step, acc_all/step


for epoch in range(100):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    vis.line(Y=np.column_stack(np.array([train_loss, train_acc, test_loss, test_acc])), X=np.column_stack(
        np.array([epoch+1, epoch+1, epoch+1, epoch+1])), win='cifar', update='append', opts=dict(legend=['train_loss', 'train_acc', 'test_loss', 'test_acc'], showlegend=True))
