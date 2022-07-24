import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Model(nn.Module):
    """
    from: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99
    """

    def __init__(self, nc):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        # for 64 x 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.conv4(x)  # for 64 x 64

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def test(network, test_loader, criterion, device):
    network.eval()
    test_loss = 0
    acc = 0

    if device.type == 'cuda':
        type_ = torch.cuda.FloatTensor
    else:
        type_ = torch.FloatTensor

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += criterion(output, target).item()

            pred = torch.exp(output)
            top_p, top_class = pred.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            acc += torch.mean(equals.type(type_))

            # pred = output.data.max(1, keepdim=True)[1]
            # acc += pred.eq(target.data.view_as(pred)).mean()

    return test_loss / len(test_loader), acc / len(test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="Epochs to train [25]", type=int, default=100)
    parser.add_argument("--batch_size", help="The size of batch images [256]", type=int, default=256)
    parser.add_argument("--learning_rate", help="Learning rate for the optimizer [0.001]", type=float, default=1e-4)
    parser.add_argument("--log_dir", help="Directory name to save the checkpoints and logs [log_dir]",
                        type=str, default="./log_dir_classifier")
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--dataset', default='mnist', help='Test model')
    parser.add_argument("--model", type=str, default=None, help="Path to model used for prediction.")
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    args = parser.parse_args()

    print(args)

    args.momentum = 0.5

    random_seed = 1234
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)


    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
            transform=torchvision.transforms.Compose([
               torchvision.transforms.Resize((64, 64)),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(
                   (0.1307,), (0.3081,))
            ]))
        args.nc = 1
        test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=True,
            transform=torchvision.transforms.Compose([
               torchvision.transforms.Resize((64, 64)),
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(
                   (0.1307,), (0.3081,))
            ]))
    if args.dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            "./data/",
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(args.imageSize),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        test_dataset = torchvision.datasets.SVHN(
            "./data/",
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(args.imageSize),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        args.nc = 3

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Model(args).to(args.device)
    criterion = nn.NLLLoss()

    if args.model is not None:
        if args.device.type == "cpu":
            network.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        else:
            network.load_state_dict(torch.load(args.model))

    if args.test:
        loss_test, acc_test = test(network, test_loader, criterion, args.device)
        print('Loss test: {:.5f}, acc test: {:.3f}'.format(loss_test, acc_test))
    else:
        # stop when the validation loss does not improve for 10 iterations to prevent overfitting
        early_stop_counter = 10

        optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

        writer = SummaryWriter(logdir=args.log_dir)

        if args.device.type == 'cuda':
            type_ = torch.cuda.FloatTensor
        else:
            type_ = torch.FloatTensor

        counter = 0
        best_loss_test = float('Inf')
        trn_tqdm = tqdm(range(args.epochs), desc="Loss: None, acc: None")
        for epoch in trn_tqdm:
            acc_train = 0
            loss_train = 0.

            network.train()

            batch_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc='Batchs', leave=True)
            for i, (data, target) in batch_tqdm:
                data = data.to(args.device)
                target = target.to(args.device)
                optimizer.zero_grad()
                output = network(data)

                # _, pred = torch.max(output.data, 1)
                # # pred = output.data.max(1, keepdim=True)[1]
                # current_acc = (pred == target).mean().item()
                # # current_acc = pred.eq(target.data.view_as(pred)).mean()

                pred = torch.exp(output)
                top_p, top_class = pred.topk(1, dim=1)
                equals = top_class == target.view(*top_class.shape)
                # current_acc = torch.mean(equals.type(torch.cuda.FloatTensor))
                current_acc = torch.mean(equals.type(type_))

                acc_train += current_acc

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                loss_train += loss.item()

                writer.add_scalar('loss', loss.item(), i + len(train_loader) * epoch)
                writer.add_scalar('acc', current_acc, i + len(train_loader) * epoch)

            torch.save(network.state_dict(), '{}/model_{:d}.pth'.format(args.log_dir, epoch + 1))
            loss_train /= len(train_loader)
            acc_train /= len(train_loader)
            loss_test, acc_test = test(network, test_loader, criterion, args.device)

            trn_tqdm.set_description_str('Loss train: {:.5f}, acc train: {:.3f}, '
                                         'loss test: {:.5f}, acc test: {:.3f}'.format(loss_train, acc_train,
                                                                                      loss_test, acc_test))

            if loss_test < best_loss_test:
                best_loss_test = loss_test
                counter = 0
                best_model = copy.deepcopy(network.state_dict())
            else:
                counter += 1
                print('Validation loss has not improved since: {:.5f}..'.format(best_loss_test),
                      'Count: ', str(counter))
                if counter >= early_stop_counter:
                    print('Early Stopping Now!!!!')
                    network.load_state_dict(best_model)
                    torch.save(network.state_dict(), '{}/best_model.pth'.format(args.log_dir))
                    break


if __name__ == '__main__':
    main()
