import argparse

import numpy as np
import torch
import tqdm as tqdm
import matplotlib.pyplot as plt

from interpolation_model import Interpolation, InterpolationLoss
from mnist_model import classifier_mnist, dcgan

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='mnist | celebA')
parser.add_argument("--num_iter", type=int, default=200, help="number of epochs of training")
parser.add_argument('--latent_dim', type=int, default=2, help='Dimensional of latent space')
parser.add_argument('--netG', default='mnist_model/netG_epoch_25.pth', help="path to netG (to continue training)")
parser.add_argument('--num_points', type=int, default=20, help='Number of points in path')
parser.add_argument('--epsilon', type=float, default=0.1, help="eps /2 + (1 - eps) * ri")
parser.add_argument('--classifier_model', type=str, default='mnist_model/model_10.pth', help="Path to classifier model.")
parser.add_argument('--favored_class', type=int, default=8, help="Class that should be favoured on interpolation path")
parser.add_argument('--use_cuda', default=True)
opt = parser.parse_args()
opt.nz = opt.latent_dim
opt.ngpu = 1
if opt.dataset == 'mnist':
    opt.nc = 1
else:
    opt.nc = 3

use_cuda = opt.use_cuda

device = torch.device("cuda:0" if use_cuda else "cpu")

# Load generator network
netG = dcgan.Generator(opt).to(device)
netG.load_state_dict(torch.load(opt.netG))

# Load classifier network
net_classifier = classifier_mnist.Model(opt.nc).to(device)
net_classifier.load_state_dict(torch.load(opt.classifier_model))
latent_dim = opt.latent_dim

dim_layers = [100, 100, 100, 100, 100, latent_dim]  # define layers sizes interpolation model
endpoint_0 = torch.rand(latent_dim).to(device)  # define first endpoint location
endpoint_1 = torch.rand(latent_dim).to(device)  # define first endpoint location

interpolation_model = Interpolation(dim_layers, endpoint_0, endpoint_1, outscale=1).cuda()

epsilon = opt.epsilon
num_points = opt.num_points
class_num = opt.favored_class
criterion = InterpolationLoss(None, device, epsilon, net_classifier=net_classifier, class_num=class_num)
optimizer = torch.optim.Adam(interpolation_model.parameters(), lr=0.001, betas=(0.5, 0.999))

values_history = []
loss_history = []

epoch_tqdm = tqdm.tqdm(range(opt.num_iter), desc="Loss: inf")
for i in epoch_tqdm:
    interpolation_model.zero_grad()
    batch = torch.rand(num_points, 1, dtype=torch.float32, device=device, requires_grad=False)
    sorted_batch, indices = batch.sort(dim=0, descending=False)
    sorted_batch[0] = 0
    sorted_batch[-1] = 1
    path = criterion.loss_new(netG, interpolation_model, sorted_batch, "l2")
    loss, ri, norm, z, values = criterion.loss_new(netG, interpolation_model, sorted_batch, "l2")
    values_history.append(values)
    loss_history.append(loss.item())
    epoch_tqdm.set_description("Loss: {:.5f}".format(loss.item()))
    loss.backward()
    optimizer.step()
    sorted_batch_test = torch.linspace(0, 1, 500, dtype=torch.float32, device=device, requires_grad=False)


batch = torch.arange(0, 1.05, 0.05, dtype=torch.float32, device=device, requires_grad=False)
batch = batch.view(-1, 1)
optimized_path, _ = interpolation_model(batch)


linear_path = endpoint_1*batch + endpoint_0*(1-batch)
linear_images = netG(linear_path).detach().cpu().numpy()
linear_images_concat = np.concatenate(linear_images, axis=2).squeeze()

optimized_images = netG(optimized_path).detach().cpu().numpy()
optimized_images_concat = np.concatenate(optimized_images, axis=2).squeeze()

plt.figure(figsize=(42, 4))
plt.imshow(linear_images_concat)
plt.title('Linear interpolation')
plt.show()

plt.figure(figsize=(42, 4))
plt.imshow(optimized_images_concat)
plt.title('Optimized interpolation')
plt.show()

optimized_path = optimized_path.cpu().detach().numpy()
linear_path = linear_path.cpu().detach().numpy()

plt.plot(optimized_path[:,0], optimized_path[:,1], 'o', label='optimized')
plt.plot(linear_path[:,0], linear_path[:,1], 'o', label='linear')
plt.legend()
plt.title('Linear vs optimized path in latent')
plt.show()


