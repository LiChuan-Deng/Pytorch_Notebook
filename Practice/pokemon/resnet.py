import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        :param stride:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # element-wise add: [b, ch_in, h, w] vs [b, ch_out, h, w]
        out = self.extra(x) + out
        out = F.relu(out)

        return out



class ResNet18(nn.Module):

    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(16),
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(16, 32, stride=3)
        # [b, 128, h, w] => [b, 256, h ,w]
        self.blk2 = ResBlk(32, 64, stride=3)
        # [b, 256, h, w] => [b, 512, h ,w]
        self.blk3 = ResBlk(64, 128, stride=3)
        # [b, 512, h, w] => [b, 1024, h ,w]
        self.blk4 = ResBlk(128, 256, stride=2)

        self.outlayer = nn.Linear(256*3*3, num_class)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv: ', x.shape) # [b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        # x = F.adaptive_avg_pool2d(x, [1,1])
        # print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x

def main():
    blk = ResBlk(64,128)
    tmp = torch.randn(2,64,224,224)
    out = blk(tmp)
    print('block: ', out.shape)

    x = torch.randn(2,3,224,224) ## the size in paper is 224*224
    model = ResNet18(5)
    out = model(x)
    print('model: ', out.shape)

    p = sum(map(lambda p:p.numel(), model.parameters()))
    print('parameters size:', p)



if __name__ == '__main__':
    main()