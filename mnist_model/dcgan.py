import argparse
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import datasets
from torchvision.utils import make_grid
from tqdm import tqdm

import utils


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def loop_save_image(tensor, dirname, normalize=False, range_interval=None, scale_each=False, start_idx=0):
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range_interval is not None:
            assert isinstance(range_interval, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range_interval):
            if range_interval is not None:
                norm_ip(t, range_interval[0], range_interval[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range_interval)
        else:
            norm_range(tensor, range_interval)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

    for i in range(ndarr.shape[0]):
        im = Image.fromarray(ndarr[i])
        im.save('{}/{:0>6d}.png'.format(dirname, start_idx + i))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def random_noise(batchSize, latentSize, device_name, type_distribution='normal', mu=1., sigma=0.2):
    if type_distribution == 'circle':
        assert latentSize == 2, 'Dimension of latent space should be equel 2 for "circle" type of latent.'
        r = sigma * torch.randn(batchSize, 1, device=device_name) + mu
        phi = 2 * math.pi * torch.rand(batchSize, 1, device=device_name)

        x = r * torch.cos(phi).to(device_name)
        y = r * torch.sin(phi).to(device_name)
        data = torch.cat((x, y), 1)
        return data.view(batchSize, latentSize, 1, 1)
    elif type_distribution == 'normal':
        return torch.randn(batchSize, latentSize, 1, 1, device=device_name)
    else:
        if latentSize > 2:
            mean = np.zeros((utils.mean_tmp.shape[0], latentSize), dtype=np.float32)
            cov = np.zeros((utils.mean_tmp.shape[0], latentSize, latentSize), dtype=np.float32)

            for i in range(utils.mean_tmp.shape[0]):
                mean[i, :2] = utils.mean_tmp[i]
                np.fill_diagonal(cov[i], utils.scale)
                cov[i, :2, :2] = utils.cov_tmp[i]
        else:
            mean = utils.mean_tmp
            cov = utils.cov_tmp

        n_samples = np.random.choice(mean.shape[0], batchSize)
        n_samples = Counter(n_samples)

        X = []
        for idx in range(mean.shape[0]):
            f = MultivariateNormal(loc=torch.from_numpy(mean[idx]), covariance_matrix=torch.from_numpy(cov[idx]))
            X.append(f.sample((n_samples[idx],)))
        X = torch.cat(X, dim=0).to(device_name)

        # f = Normal(torch.tensor([0.0], dtype=torch.float32, device=device_name),
        #            torch.tensor([0.01], dtype=torch.float32, device=device_name))
        # X = torch.cat((X.to(device_name),
        #                f.sample([batchSize, latentSize - 2]).squeeze()),
        #               dim=1).view(batchSize, latentSize, 1, 1)
        return X.view(batchSize, latentSize, 1, 1)


def train(opt):
    current_date = datetime.now()
    current_date = current_date.strftime('%d%b_%H%M%S')
    str_params = '------------------------------\nParameters:\n'
    for arg in vars(opt):
        str_params += '{}: {}\n'.format(arg, getattr(opt, arg))
    str_params += '------------------------------\n'
    print(str_params)

    log_dir = '{}/logs/{}'.format(opt.outf, current_date)
    model_dir = '{}/models/{}'.format(opt.outf, current_date)
    img_dir = '{}/images/{}'.format(opt.outf, current_date)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    with open('{}/params.txt'.format(model_dir), 'w') as f:
        f.write(str_params)
    del str_params

    if opt.dataset == 'mnist':
        dataset = datasets.MNIST(
            "{}/mnist".format(opt.data_dir),
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
        opt.nc = 1
    elif opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            "{}/cifar10".format(opt.data_dir),
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        opt.nc = 3
    elif opt.dataset == 'celebA':
        celebA_dir = "{}/celebA/".format(opt.data_dir)
        utils.download_celeb_a(celebA_dir)

        dataset = datasets.ImageFolder(
            root=celebA_dir,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        opt.nc = 3

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, drop_last=True, num_workers=int(opt.workers))

    netG = Generator(opt).to(opt.device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        netG.eval()
    # print(netG)

    netD = Discriminator(opt).to(opt.device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
        netD.eval()
    # print(netD)

    digits = int(math.log10(opt.niter)) + 1
    np.savez('{}/densities.npz'.format(model_dir), scale=utils.scale, mean=utils.mean_tmp, cov=utils.cov_tmp, nz=opt.nz)

    criterion = nn.BCELoss()

    fixed_noise = random_noise(opt.batchSize, opt.nz, device_name=opt.device, type_distribution=opt.randomLatent,
                               mu=opt.mu_sigma[0], sigma=opt.mu_sigma[1])
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    writer = SummaryWriter(log_dir=log_dir)

    num_iters = len(dataloader)

    batch_images_visible = True
    epoch_images_visible = True

    for epoch in tqdm(range(1, opt.niter + 1)):
        batch_tqdm = tqdm(enumerate(dataloader, 0), desc="Batches", leave=False, total=num_iters)
        for i, data in batch_tqdm:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(opt.device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=opt.device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = random_noise(batch_size, opt.nz, device_name=opt.device, type_distribution=opt.randomLatent,
                                 mu=opt.mu_sigma[0], sigma=opt.mu_sigma[1])
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            batch_tqdm.set_description("Batches Loss_D: {:.4f} Loss_G: {:.4f} D(x): "
                                       "{:.4f} D(G(z)): {:.4f}/{:.4f}".format(errD.item(), errG.item(),
                                                                              D_x, D_G_z1, D_G_z2))

            index = epoch * num_iters + i
            writer.add_scalar('loss/D', errD.item(), index)
            writer.add_scalar('loss/G', errG.item(), index)
            writer.add_scalar('discriminator/D_x', D_x, index)
            writer.add_scalar('discriminator/D_G_z1', D_G_z1, index)
            writer.add_scalar('discriminator/D_G_z2', D_G_z2, index)

            if i % 100 == 0:
                fake = netG(fixed_noise)

                tmp = math.ceil(math.sqrt(opt.batchSize))
                try:
                    del batch_images_visible
                    writer.add_image('images_real/bs{}_nz{}'.format(opt.batchSize, opt.nz),
                                     make_grid(((real_cpu + 1) / 2).data, nrow=tmp), index)
                except NameError:
                    pass
                writer.add_image('images_fake/bs{}_nz{}'.format(opt.batchSize, opt.nz),
                                 make_grid(((fake + 1) / 2).data, nrow=tmp), index)

        # do checkpointing and images
        # if epoch % 5 == 0:
        if True:
            try:
                del epoch_images_visible
                save_image(real_cpu, '{}/real_samples.png'.format(img_dir), normalize=True)
            except NameError:
                pass
            save_image(fake.detach(),
                       '{}/fake_samples_epoch_{:0{width}d}.png'.format(img_dir, epoch, width=digits),
                       normalize=True)

            torch.save(netG.state_dict(), '{}/netG_epoch_{:d}.pth'.format(model_dir, epoch))
            torch.save(netD.state_dict(), '{}/netD_epoch_{:d}.pth'.format(model_dir, epoch))


def test(opt):
    if opt.dataset == 'mnist':
        opt.nc = 1
    elif opt.dataset == 'cifar10':
        opt.nc = 3
    elif opt.dataset == 'celebA':
        opt.nc = 3

    netG = Generator(opt).to(opt.device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    netG.eval()
    # print(netG)

    # freeze module
    for param in netG.parameters():
        param.requires_grad = False

    Path(opt.outf).mkdir(parents=True, exist_ok=True)
    paths = np.load(opt.path2way)

    digits = int(math.ceil(math.log10(len(paths.files))))
    nrow, ncol = paths[paths.files[0]].shape[:2]
    # nrow = int(math.ceil(math.sqrt(nrow)))

    with torch.no_grad():
        for i, v in tqdm(enumerate(paths.files), total=len(paths.files)):
            noise = torch.from_numpy(paths[v]).to(opt.device)
            fake = netG(noise)

            filename = '{}/{:0{width}d}.png'.format(opt.outf, i, width=digits)
            save_image(fake.detach(), filename, nrow=nrow, normalize=True, scale_each=True)


def test_disc(opt):
    if opt.dataset == 'mnist':
        opt.nc = 1
    elif opt.dataset == 'cifar10':
        opt.nc = 3
    elif opt.dataset == 'celebA':
        opt.nc = 3

    netG = Generator(opt).to(opt.device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    netG.eval()
    # print(netG)

    # freeze module
    for param in netG.parameters():
        param.requires_grad = False

    netD = Discriminator(opt).to(opt.device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    netD.eval()
    # print(netD)

    # freeze module
    for param in netD.parameters():
        param.requires_grad = False

    Path(opt.outf).mkdir(parents=True, exist_ok=True)
    nrow = int(math.ceil(math.sqrt(opt.batchSize)))

    print('Sigma:', opt.sigma)
    with torch.no_grad(), open('{}/discriminator.csv'.format(opt.outf), 'w') as f:
        f.write('sigma;min;median;mean;max\n')
        for sigma in opt.sigma:
            torch.manual_seed(opt.manualSeed)
            output = []
            f.write('{};'.format(sigma))
            for i in tqdm(range(opt.niter)):
                noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=opt.device)
                norm_noise = torch.norm(noise, dim=1).unsqueeze(-1)
                noise = noise / norm_noise
                if sigma != 1.:
                    noise *= sigma
                fake = netG(noise)
                output.append(netD(fake.detach()).cpu().numpy())

                if i == 0:
                    save_image(fake.detach(),
                               '{}/fake_samples_{}.png'.format(opt.outf, sigma),
                               nrow=nrow, normalize=True, scale_each=True)
            output = np.concatenate(output, axis=0)
            f.write('{:.8f};{:.8f};{:.8f};{:.8f}\n'.format(np.min(output), np.median(output), np.mean(output),
                                                           np.max(output)))


def test_gen(opt):
    if opt.dataset == 'mnist':
        opt.nc = 1
    elif opt.dataset == 'cifar10':
        opt.nc = 3
    elif opt.dataset == 'celebA':
        opt.nc = 3

    netG = Generator(opt).to(opt.device)
    netG.apply(weights_init)

    if opt.netG != '':
        if opt.device.type == "cpu":
            netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
        else:
            netG.load_state_dict(torch.load(opt.netG))
    netG.eval()
    # print(netG)

    # freeze module
    for param in netG.parameters():
        param.requires_grad = False

    Path(opt.outf).mkdir(parents=True, exist_ok=True)
    # nrow = int(math.ceil(math.sqrt(opt.batchSize)))
    #
    # print('Sigma:', opt.sigma)
    # with torch.no_grad():
    #     if opt.use_mu:
    #         mu = torch.randn(1, opt.nz, 1, 1, device=opt.device)
    #
    #     for sigma in opt.sigma:
    #         torch.manual_seed(opt.manualSeed)
    #
    #         fake_dir = '{}/{}'.format(opt.outf, sigma)
    #
    #         Path(fake_dir).mkdir(parents=True, exist_ok=True)
    #
    #         for i in tqdm(range(opt.niter)):
    #             noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=opt.device)
    #             norm_noise = torch.norm(noise, dim=1).unsqueeze(-1)
    #             noise = noise / norm_noise
    #             if opt.use_mu:
    #                 if sigma != 1.:
    #                     noise = sigma * noise + mu
    #                 else:
    #                     noise = noise + mu
    #             else:
    #                 if sigma != 1.:
    #                     noise = sigma * noise
    #             fake = netG(noise)
    #
    #             loop_save_image(fake.detach(), fake_dir, normalize=True, scale_each=True, start_idx=i*opt.batchSize)
    #
    #             if i == 0:
    #                 save_image(fake.detach(),
    #                            '{}/fake_samples_{}.png'.format(opt.outf, sigma),
    #                            nrow=nrow, normalize=True, scale_each=True)

    with torch.no_grad():
        torch.manual_seed(opt.manualSeed)

        noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=opt.device)
        norm_noise = torch.norm(noise, dim=1).unsqueeze(-1)
        noise = noise / norm_noise
        for i in tqdm(range(opt.batchSize)):
            z = torch.from_numpy(np.array(opt.sigma).reshape([-1, 1, 1, 1]).astype(np.float32)).to(device=opt.device) * \
                noise[i]
            fake = netG(z)
            save_image(fake.detach(), '{}/fake_samples_{}.png'.format(opt.outf, i),
                       nrow=len(opt.sigma), normalize=True, scale_each=True)


def main():
    parser = argparse.ArgumentParser()
    # parser.register('type', 'bool', str2bool)  # add type keyword to registries
    parser.add_argument('--dataset', required=True, help='mnist | cifar10 | celebA')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)  # 128
    parser.add_argument('--ndf', type=int, default=64)  # 128
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
    parser.add_argument('--data-dir', default='./data', help='folder to data sets')
    parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')
    # parser.add_argument('--randomLatent', required=True, type=str2bool, help='True -> randn, False -> semicircle')
    parser.add_argument('--randomLatent', default='normal', help='Choose from: normal, circle, semicircle')
    parser.add_argument('--path2way', default='./logs/dim20/06Mar_090126-dim20/paths.npz',
                        help='Path to file with way in latent space')
    parser.add_argument('--test', action='store_true', help='Only test model')
    parser.add_argument('--sigma', nargs='+', type=float, default=[1.],
                        help='The standard deviation for a normal distribution (distribution of latent space).')
    parser.add_argument('--use_mu', action='store_true', help='Use mean from latent')
    parser.add_argument('--mu_sigma', nargs='+', type=float, default=[1., 0.2])

    opt = parser.parse_args()

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        opt.cuda = True
        print("Used CUDA device")

    opt.device = torch.device("cuda:0" if opt.cuda else "cpu")

    if opt.test:
        print('Test process.')
        # test(opt)
        # test_disc(opt)
        test_gen(opt)
    else:
        print('Start to train model.')
        train(opt)


if __name__ == '__main__':
    main()
