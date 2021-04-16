import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
import random
from matplotlib import pyplot as plt
h_dim = 400
batchsz = 512
viz = visdom.Visdom()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z: [b, 2] => [b, 2]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )

    def forward(self, z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def data_generator():
    """
    8-gaussian mixture models
    :return:
    """
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []

        for i in range(batchsz):

            point = np.random.randn(2) *0.02
            center = random.choice(centers)
            # N(0,1) + center x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset

def generate_image(D, G, xr, epoch):

    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1,2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points) # [16384, 2]
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)

    # draw samples
    with torch.no_grad():
        z = torch.randn(batchsz, 2) # [b, 2]
        samples = G(z).cpu().numpy() # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d'%epoch))


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr:
    :param xf:
    :return:
    """
    LAMBDA = 0.3

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(batchsz, 1)
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp

def main():

    torch.manual_seed(24)
    np.random.seed(24)

    data_iter = data_generator()
    # x = next(data_iter)
    # [b, 2]
    # print(x.shape)

    G = Generator()
    D = Discriminator()
    # G.apply(weights_init)
    # D.apply(weights_init)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5,0.9))
    optim_D = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5,0.9))

    viz.line([[0,0]], [0], win='loss', opts=dict(title='loss',legend=['D', 'G']))

    for epoch in range(50000):

        # 1. train Discriminator firstly
        for _ in range(5):
            # 1.1 train on real data
            xr = next(data_iter)
            xr = torch.from_numpy(xr)
            # [b, 2] => [b, 1]
            predr = D(xr)
            lossr = -(predr.mean())

            # 1.2 train on fake data
            z = torch.randn(batchsz, 2)
            xf = G(z).detach()  # tf.stop_gradient()
            predf = D(xf)
            lossf = predf.mean()

            gp = gradient_penalty(D, xr, xf)

            # aggregate all
            loss_D = lossr + lossf + gp

            # optimzie
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batchsz, 2)
        xf = G(z)
        predf = D(xf)
        loss_G = -(predf.mean())

        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

            print(loss_D.item(), loss_G.item())

            generate_image(D, G, xr, epoch)








if __name__ == '__main__':
    main()
