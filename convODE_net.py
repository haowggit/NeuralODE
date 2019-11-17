import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from ode_solver import odeint
from ode_solver import adjoint_method



class neural_ode_func(nn.Module):
    def __init__(self, func):
        super(neural_ode_func, self).__init__()
        assert isinstance(func, nn.Module)
        self. func = func

    def forward(self, x0, t=torch.tensor([0., 1.]), h=None):
        f_params = self.func.parameters()
        flat_params = torch.cat([p_.flatten() for p_ in f_params])
        x = adjoint_method.apply(self.func, h, x0, t, flat_params)
        return x


def conv3(in_dim, out_dim, ksize=3, stride=1, padding=0, bias=True):
    return nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=1, bias=bias)


def norm(dim):
    return nn.BatchNorm2d(dim)


def movetime(x, t):

    tt = torch.ones_like(x[:, :1, :, :]) * t
    ttx = torch.cat([tt, x], 1)
    return ttx


class convODE(nn.Module):
    def __init__(self, dim):
        super(convODE, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm2 = norm(dim)
        self.conv1 = conv3(dim+1, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = conv3(dim+1, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, x, t):
        self.nfe += 1
        x_out = self.norm1(x)
        x_out = self.relu(x_out)
        x_out = movetime(x_out, t)
        x_out = self.conv1(x_out)
        x_out = self.norm2(x_out)
        x_out = self.relu(x_out)
        x_out = movetime(x_out, t)
        x_out = self.conv2(x_out)
        x_out = self.norm3(x_out)
        return x_out


class Trainer(object):
    def __init__(self, model, model_name, nepochs, batch_size, test_batch_size):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(
            model.parameters(), lr=0.005, weight_decay=1e-7)
        self.criterion = nn.CrossEntropyLoss()
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        def get_dataset(batch_size=64, test_batch_size=256):
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_set = datasets.MNIST(
                root='.data/MNIST', train=True, download=True, transform=transform)
            test_set = datasets.MNIST(
                root='.data/MNIST', train=False, download=True, transform=transform)
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(
                train_set, batch_size=test_batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(
                test_set, batch_size=test_batch_size, shuffle=True, num_workers=4)
            return train_set, test_set, train_loader, val_loader, test_loader

        self.train_set, self.test_set, self.train_loader, self.val_loader, self.test_loader = get_dataset(
            batch_size=self.batch_size, test_batch_size=self.test_batch_size)
        self.batches_per_epoch = len(self.train_loader)
        self._reset()

    def __test__(self):
        self.model(next(iter(self.train_loader))[0].cuda())




    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def one_hot(self, x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

    def get_accuracy(self, dataset_loader):
        total_correct = 0

        for x, y in dataset_loader:
            x = x.to(self.device)
            y = self.one_hot(np.array(y.numpy()), 10)
            target_class = np.argmax(y, axis=1)
            predicted_class = np.argmax(
                self.model(x).cpu().detach().numpy(), axis=1)
            total_correct += np.sum(predicted_class == target_class)
        return total_correct / len(dataset_loader.dataset)

    def get_accu(self, DataLoader):
        self.model.eval()
        correct_count = 0.0
        Iterator = iter(DataLoader)
        count = 0
        while True:

            if(count == DataLoader.__len__()):
                break

            try:
                x, y = next(Iterator)

            except StopIteration:
                Iterator = iter(DataLoader)
                x, y = next(Iterator)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            outs = self.model(x)
            _, pred = torch.max(outs, -1)
            eqaul = pred == y
            correct_count += torch.sum(eqaul.float()).item()

            count += 1

        return correct_count / len(DataLoader.dataset)

    def train_network(self):
        if not os.path.exists('./'+self.model_name):
            os.makedirs('./'+self.model_name)

        self.model.train()
        start = time.time()
        sum_loss = 0
        train_iterator = iter(self.train_loader)

        for i in range(self.nepochs*self.batches_per_epoch):
            if i % 20 == 0:
                print('iteration: ' + str(i))

            self.optimizer.zero_grad()
            try:
                images, labels = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                images, labels = next(train_iterator)

            images = images.to(self.device)
            labels = labels.to(self.device)
            print(images.shape)
            forwardpass = self.model(images)
            # calculate the loss between predicted and target keypoints
            loss = self.criterion(forwardpass, labels)
            self.loss_history.append(loss)
            loss.backward()
            sum_loss += loss.item()
            # update the weights
            self.optimizer.step()

            if i % self.batches_per_epoch == 0:
                avg_loss = sum_loss/self.batches_per_epoch
                end = time.time()
                validation_time_start = time.time()
                val_acc = self.get_accu(self.val_loader)
                train_acc = self.get_accu(self.train_loader)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                validation_time_end = time.time()
                print(str(validation_time_end-validation_time_start))
                log = 'Epoch: {}, iteration: {}  Avg. Loss: {},  Elapsed time: {}, Accuracy: {} ({})  '.format(
                    self.epoch, i+1, str(avg_loss), str(end-start), train_acc, val_acc)
                print(log)
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epapsed time': end-start,
                    'loss': avg_loss,
                    'finish': False

                }, os.path.join('./resnet', 'model.pth'))
                start = time.time()
                sum_loss = 0
                self.epoch += 1
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epapsed time': end-start,
            'loss': avg_loss,
            'finish': True
        }, os.path.join('./resnet', 'model.pth'))

    def visualize(self):
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(self.loss_history, 'o')
        plt.xlabel('Iteration')

        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(self.train_acc_history, '-o', label='train')
        plt.plot(self.val_acc_history, '-o', label='val')
        plt.plot([0.5] * len(self.val_acc_history), 'k--')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15, 12)
        plt.show()

    def get_model(self):
        return self.model

    def test_model(self):
        self.model.eval()

        iterator = iter(self.test_loader)
        for i in range(np.random.choice(len(self.test_loader))):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(self.test_loader)
                images, labels = next(iterator)

        images = images.to(self.device)
        outputs = self.model(images)
        predictions = torch.argmax(outputs, axis=1)

        grid = torchvision.utils.make_grid(images.cpu(), nrow=64)
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid, (1, 2, 0)))

        print(predictions.cpu().numpy())
        print(labels.cpu().numpy())
        plt.show()

    def test_acc(self):
        self.model.eval()
        return self.get_accu(self.test_loader)

    def load_checkpoint(self, path):

        checkpoint = torch.load(path)

        self.model = self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        print(self.model)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


if __name__ == '__main__':
    downsampling_layers = [
        nn.Conv2d(in_channels=1, out_channels=64,
                  kernel_size=3, stride=1),
        nn.GroupNorm(min(32, 64), 64),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64,
                  kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(min(32, 64), 64),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64,
                  kernel_size=4, stride=2, padding=1)
    ]

    func=convODE(64)

    feature_layers = [neural_ode_func(func)]
    classify_layers = [nn.GroupNorm(min(32, 64), 64), nn.ReLU(
        inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    convode_net = nn.Sequential(*downsampling_layers, *feature_layers, *classify_layers)
    trainer = Trainer(convode_net, 'convode', nepochs=4,batch_size=128,test_batch_size=256)
    trainer.__test__()
    
    
    trainer.train_network()
    trainer.visualize()
    trainer.test_model()
    print(trainer.test_acc())


