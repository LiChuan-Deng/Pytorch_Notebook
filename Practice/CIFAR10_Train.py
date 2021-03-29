import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms
from lenet5 import Lenet5

def main():
    batch_size = 32

    cifar_train = datasets.CIFAR10('cifar_data', True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar_data', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):

        # train
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32], [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch,loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32], [b]
                x, label = x.to(device), label.to(device)
                # [b,10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print(epoch, acc)


if __name__ == '__main__':
    main()