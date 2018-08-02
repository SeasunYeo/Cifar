import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.nn import functional
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.func1 = nn.Linear(16*5*5, 120)
        self.func2 = nn.Linear(120, 84)
        self.func3 = nn.Linear(84, 10)

    def forward(self, x):
        x = functional.max_pool2d(functional.relu(self.conv1(x)), (2, 2))
        x = functional.max_pool2d(functional.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = functional.relu(self.func1(x))
        x = functional.relu(self.func2(x))
        x = self.func3(x)
        return x


def getData():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True)
    testset = torchvision.datasets.CIFAR10(
        './data/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


if torch.cuda.is_available():
    print("you have a gpu")
    device = 'cuda'
else:
    print("you should buy a gpu")
    device = 'cpu'


def trainModel():
    trainloader, testloader, _ = getData()
    net = Net()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    for epoch in range(10):
        for step, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if step % 1000 == 999:
                acc = testNet(net, testloader)
                print('Epoch:', epoch, '|test accurcy:%.4f' % acc)
    print('Finished Training')
    return net


def testNet(net, testloader):
    correct, total = .0, .0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return float(correct)/total


net = trainModel()
