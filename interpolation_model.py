from collections import OrderedDict
import torch.nn as nn
import torch


class Interpolation(nn.Module):
    def __init__(self, dim_layers, endpoint1, endpoint2, tanh=True, outscale=1):
        super(Interpolation, self).__init__()
        assert isinstance(dim_layers, list), 'Type of parameter "dim_layers" should be list.'
        self.endpoint1 = endpoint1
        self.endpoint2 = endpoint2
        self.outscale = outscale

        self.net = nn.Sequential(
            OrderedDict([
                ('dense0', nn.Linear(1, dim_layers[0]))
            ]))

        for i in range(1, len(dim_layers)):
            self.net.add_module('relu{:d}'.format(i - 1), nn.ReLU(True))
            self.net.add_module('dense{:d}'.format(i), nn.Linear(dim_layers[i - 1], dim_layers[i]))
        if tanh:
            self.net.add_module('tanh', nn.Tanh())

    def forward(self, inputs):
        # assert inputs[0] == 0 and inputs[-1] == 1
        output = self.net(inputs) * self.outscale
        transformed_output = output - (inputs * output[-1] + (1 - inputs) * output[0]) + (inputs * self.endpoint2 +
                                                                                          (1 - inputs) * self.endpoint1)

        return transformed_output, output


class InterpolationLoss(object):
    def __init__(self, density, device, epsilon=0.2, net_classifier=None, class_num=0):
        self.numeric_error = 1e-20
        self.epsilon = epsilon
        self.device = device
        self.density = density

        self.net_classifier = net_classifier
        self.class_num = class_num

    def loss_new(self, generator, interpolation, path_t, norm='l2'):
        z, _ = interpolation(path_t)
        fake = generator(z)

        path_t_squeezed = path_t.squeeze()
        rolled_path_t_squeezed = torch.roll(path_t_squeezed, (1,), (0,))
        path_t_squeezed_mid = torch.div(torch.add(path_t_squeezed, rolled_path_t_squeezed), 2.)
        path_t_mid = path_t_squeezed_mid.view(-1, 1)
        z_mid, _ = interpolation(path_t_mid)
        fake_mid = generator(z_mid)
        classifier_preds_mid = self.net_classifier(fake_mid)

        classifier_preds = self.net_classifier(fake)
        values = torch.exp(classifier_preds)

        ri = values[:, self.class_num]

        values = torch.pow(values, 4 * path_t * (1 - path_t))
        rolled = torch.roll(fake, (1,), (0,))
        if norm == 'l2':
            norm_squared = torch.mean(torch.pow((fake - rolled), 2), dim=(1, 2, 3))
        else:
            norm_squared = torch.mean(torch.abs((fake - rolled)), dim=(1, 2, 3))
        path_t = path_t.squeeze()
        rolled_t = torch.roll(path_t, (1,), (0,))
        diff_t = path_t - rolled_t

        ri = self.epsilon / 2. + (1. - self.epsilon) * ri
        rolled_ri = torch.roll(ri, (1,), (0,))
        ri_mid = (ri + rolled_ri) / 2.

        log_square = -torch.log(ri_mid + self.numeric_error) * torch.sqrt(torch.pow(diff_t, 2) + norm_squared)

        return log_square[1:].sum(), ri, torch.sqrt(torch.pow(diff_t, 2) + norm_squared)[1:], z, values
