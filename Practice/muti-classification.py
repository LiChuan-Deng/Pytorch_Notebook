import torch
import torchvision
from torch.nn import functional as F

batch_size = 200
epochs = 10
learning_rate = 0.01

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])),
    batch_size = batch_size, shuffle = False)

w1, b1 = torch.randn(200,784, requires_grad=True),\
        torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200,200, requires_grad=True),\
        torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10,200, requires_grad=True),\
        torch.zeros(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

def forward(x):
    x = F.relu(x@w1.t()+b1)
    x = F.relu(x@w2.t()+b2)
    x = F.relu(x@w3.t()+b3)
    return x

optimizer = torch.optim.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)

        logits = forward(data)
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
    for data, target in test_loader:
        data = data.view(-1,28*28)
        logits = forward(data)
        test_loss +=criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.*correct / len(test_loader.dataset)))
