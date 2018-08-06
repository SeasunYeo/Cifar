import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transform
from SENet import *
from ResNet import *
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
    transform.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
])
transform_test = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=35, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)
net = SENet18()
#net = ResNet34()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05,
                      momentum=0.9, weight_decay=5e-4)


def train(epoch):
    print('Epoch:%d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets)in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('(%d/%d)Loss:%.3f | Acc: %.3f%%(%d/%d)' % (total/targets.size(0), 50000.0 /
                                                         targets.size(0), train_loss/(batch_idx+1), 100*correct/total, correct, total))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets)in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('(%d/%d)Loss:%.3f | Acc:%.3f%%(%d/%d)' % (total/targets.size(0), 10000.0 /
                                                            targets.size(0), test_loss/(batch_idx+1), 100*correct/total, correct, total))


for epoch in range(100):
    train(epoch)
    test(epoch)
