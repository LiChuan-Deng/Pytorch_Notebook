import torch
import torchvision
from torch.nn import functional as F

batch_size = 200
epochs = 10
learning_rate = 0.01

train_db = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                      ]))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)

test_db = torchvision.datasets.MNIST('mnist_data', train=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                      ]))
test_loader = torch.utils.data.DataLoader(
    test_db,
    batch_size=batch_size, shuffle=True)

print('train:', len(train_db), 'test:', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000,10000])
print('db1:', len(train_db), 'db2:', len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size, shuffle=True)

class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)
criteon = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits,target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1,28*28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss +=criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100.*correct / len(val_loader.dataset)))


test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1,28*28)
    data, target = data.to(device), target.cuda()
    logits = net(data)
    test_loss +=criteon(logits, target).item()

    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100.*correct / len(test_loader.dataset)))