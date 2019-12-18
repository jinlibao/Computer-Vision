#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageFile
from argparse import ArgumentParser
import timeit


class TorchNet(nn.Module):

    def __init__(self):
        super(TorchNet, self).__init__()
        self.name = 'TorchNet'
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.name = 'LeNet'
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class MiniVGGNet(nn.Module):

    def __init__(self):
        super(MiniVGGNet, self).__init__()
        self.name = 'MiniVGGNet'
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool(self.batch_norm1(F.relu(self.conv2(x)))))
        x = self.batch_norm2(F.relu(self.conv3(x)))
        x = self.dropout1(self.pool(self.batch_norm2(F.relu(self.conv4(x)))))
        x = x.view(-1, 8 * 8 * 64)
        x = F.relu(self.fc1(x))
        x = self.batch_norm3(x)
        x = self.dropout2(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


# Rosebrock
class ShallowNet(nn.Module):

    def __init__(self):
        super(ShallowNet, self).__init__()
        self.name = 'ShallowNet'
        self.conv = nn.Conv2d(3, 32, 3, 1, 1)
        self.fc = nn.Linear(32 * 32 * 32, 3)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 32 * 32 * 32)
        x = F.softmax(self.fc(x), dim=1)
        return x


class AnimalsDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        image= self.dataset[0][idx]
        target = self.dataset[1][idx]
        if self.transform:
            image = self.transform(image)
        return [image, target]


class CNNTrainer(object):

    def __init__(self, data_dir, model_dir, batch_size, test_size,
                 validation_size, learning_rates, momentums, epochs,
                 net_name, lr_step, lr_step_size, lr_gamma):
        # Visualization
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        plt.style.use('ggplot')
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        self.linestyle = ['-', '--', '-.', ':']

        # Preprocess images
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)), # resize all the image to 32x32x3
            transforms.ToTensor(),       # [0, 255] -> [0, 1.0], (H x W x C) -> (C x H x W)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1.0] -> [-1.0, 1.0]
        ])
        self.dataset = self.read_image(data_dir, test_size, validation_size,
                                       batch_size, self.transform)
        train_loader, validation_loader, test_loader, classes = self.dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.classes = classes

        # Tuning hyperparamters
        self.learning_rates = learning_rates
        self.momentums = momentums
        self.epochs = epochs
        self.lr_step = lr_step
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        # Choose device to train on
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        # Select CNN
        self.net_name = net_name
        self.nets = {
            'MiniVGGNet': MiniVGGNet(),
            'LeNet': LeNet(),
            'ShallowNet': ShallowNet(),
        }


    def imshow(self, image):
        image = image / 2 + 0.5

        # Method 1
        image_pil = transforms.ToPILImage()(image)
        plt.imshow(image_pil)

        # # Method 2
        # image_np = image.numpy()
        # plt.imshow(np.transpose(image_np, (1, 2, 0)))

        plt.show()

    def plot(self, df, net_name, learning_rates, momentums, plot_dir='.'):
        linestyle = self.linestyle
        for lr in learning_rates:
            for m in momentums:
                d = df[(df['lr'] == lr) & (df['momentum'] == m)]
                fig = plt.figure(figsize=(8, 6))
                plt.plot(d['epoch'], d['acc_train'], linestyle[0], label='acc_train')
                plt.plot(d['epoch'], d['acc_val'], linestyle[1], label='acc_val')
                plt.plot(d['epoch'], d['loss_train'], linestyle[2], label='loss_train')
                plt.plot(d['epoch'], d['loss_val'], linestyle[3], label='loss_val')
                plt.xlabel('epoch #', color='black')
                plt.ylabel('loss/accuracy', color='black')
                plt.title('{:s} (lr = ${:.2f}$, momentum = ${:.2f}$)'.format(
                    net_name, lr, m))
                plt.legend(loc='best')
                [i.set_color('black') for i in plt.gca().get_xticklabels()]
                [i.set_color('black') for i in plt.gca().get_yticklabels()]
                plt.show(block=False)
                filename = '{:s}/{:s}_{:.2f}_{:.2f}.pdf'.format(
                    plot_dir, net_name, lr, m
                )
                print('Saving plot to {:s}'.format(filename))
                plt.savefig(filename, bbox_inches='tight')

    def read_image(self, data_dir, test_size=0.25, validation_size=0.2,
                   batch_size=32, transform=None):
        if not transform:
            transform = self.transform
        # Read images and labels
        data, labels = [], []
        classes = [c for c in os.listdir(data_dir) if not c.startswith('.')]
        paths = [os.path.join(data_dir, c) for c in classes]
        print('Loading images...')
        for path in paths:
            files = os.listdir(path)
            filenames = [os.path.join(path, file) for file in files]
            data.extend([Image.open(f).convert('RGB') for f in filenames])
            labels.extend([classes.index(f.split('_')[0]) for f in files])

        print('Generating train_set/validation_set/test_set...')
        # Split data into train_set, validation_set, and test_set
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size, random_state=42)
        train_set = (X_train, Y_train)
        validation_set = (X_val, Y_val)
        test_set = (X_test, Y_test)

        # Transform RGB image to torch.Tensor
        validation_set = AnimalsDataset(validation_set, transform)
        train_set = AnimalsDataset(train_set, transform)
        test_set = AnimalsDataset(test_set, transform)

        # Create DataLoader for train_set, validation_set, and test_set
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        train_size = len(train_loader) * train_loader.batch_size
        test_size = len(test_loader) * test_loader.batch_size
        validation_size = len(validation_loader) * validation_loader.batch_size
        total_size = train_size + test_size + validation_size

        print('Summary:\n\t{:d} training samples'.format(train_size))
        print('\t{:d} validation samples'.format(validation_size))
        print('\t{:d} test samples'.format(test_size))
        print('\t{:d} samples in total'.format(total_size))

        return (train_loader, validation_loader, test_loader, classes)

    def get_report(self, net, data_loader):
        net = net.to(self.device)
        with torch.no_grad():
            predicts_total, targets_total = [], []
            for batch in data_loader:
                images, targets = batch
                images = images.to(self.device)
                outputs = net(images)
                _, predicts = torch.max(outputs, 1)
                predicts_total.extend(predicts.cpu().numpy())
                targets_total.extend(targets.numpy())
            confusion = confusion_matrix(targets_total, predicts_total)
            report_dict = classification_report(targets_total, predicts_total, target_names=self.classes, output_dict=True)
            report = classification_report(targets_total, predicts_total, target_names=self.classes)
        return (report_dict, report, confusion)

    def model(self, net, data_loader, criterion=None, optimizer=None, mode='eval'):
        net = net.to(self.device)
        running_loss = 0.0
        for batch in data_loader:
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            if mode == 'train':
                optimizer.zero_grad()
                outputs = net(images)
                outputs = outputs.to(self.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            else:
                outputs = net(images)
                outputs = outputs.to(self.device)
                loss = criterion(outputs, targets)
            running_loss += loss.item() / len(data_loader)
        report_dict, _, _ = self.get_report(net, data_loader)

        return (running_loss, report_dict['accuracy'])

    def CNN(self, net, dataset, epochs, lr, momentum, dest, lr_step, lr_step_size, lr_gamma) :
        train_loader, validation_loader, test_loader, classes = dataset
        # Instantiate CNN, pick loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) # lr: 1e-1, 1e-2, 1e-3, 1e-4; momentum: 0.9-0.99
        current_lr = lr

        history = []
        nets = []
        if lr_step:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        for epoch in range(epochs):
            loss_train, acc_train = self.model(net, train_loader, criterion, optimizer, 'train')
            if lr_step:
                lr_scheduler.step()
                current_lr = lr * (lr_gamma ** (epoch // lr_step_size))
            loss_val, acc_val = self.model(net, validation_loader, criterion)
            nets.append(net.state_dict())
            history.append({
                'epoch': epoch + 1,
                'acc_train': acc_train,
                'acc_val': acc_val,
                'loss_train': loss_train,
                'loss_val': loss_val,
                'lr': current_lr,
                'momentum': momentum
            })
            print('epoch [{:d}/{:d}] loss_train: {:5.3f}, acc_train: {:5.3f}, loss_val: {:5.3f}, acc_val: {:5.3f}'.format(
                epoch + 1, epochs, loss_train, acc_train, loss_val, acc_val))
        print('Finished training.')

        idx = np.argmax(np.array(pd.DataFrame(history)['acc_val']))
        print('The best epoch is {:d}'.format(idx + 1))

        if not os.path.exists(dest):
            os.mkdir(dest)
        model_filename = '{:s}/{:s}_{:.2f}_{:.2f}_{:d}.pth'.format(dest, net.name, lr, momentum, epochs)
        print('Saving trained model to {:s}'.format(model_filename))
        torch.save(nets[idx], model_filename)

        print('Evaluating network at best epoch...')
        net.load_state_dict(nets[idx])
        # Calculate the accuracy and generate classification report
        report_dict, report, confusion = self.get_report(net, test_loader)

        return (report_dict, report, confusion, history)

    def test(self):
        train_loader, validation_loader, test_loader, classes = self.dataset
        net = self.nets[self.net_name]
        epochs = self.epochs
        learning_rates = self.learning_rates
        momentums = self.momentums
        dataset = self.dataset
        model_dir = self.model_dir
        lr_step = self.lr_step
        lr_step_size = self.lr_step_size
        lr_gamma = self.lr_gamma

        if self.use_cuda:
            print('Training {:s} on GPU...'.format(net.name))
        else:
            print('Training {:s}...'.format(net.name))
        all_history = []
        for lr in learning_rates:
            for momentum in momentums:
                dest='{:s}/{:s}'.format(model_dir, net.name)
                print('learning rate: {:e}, momentumn: {:e}'.format(lr, momentum))
                report_dict, report, confusion, history = self.CNN(net, dataset, epochs, lr, momentum, dest, lr_step, lr_step_size, lr_gamma)
                pd.DataFrame(report_dict).T.to_csv('{:s}/{:s}_{:.2f}_{:.2f}_{:d}_report.csv'.format(dest, net.name, lr, momentum, epochs))
                all_history.extend(history)
                print('Confusion matrix:\n', confusion)
                print('Classification report:\n', report)
        df = pd.DataFrame(all_history)
        df.to_csv('{:s}/training_history.csv'.format(dest))
        self.plot(df, net.name, learning_rates, momentums, dest)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', help='Root directory of animals dataset')
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='Folder to store the trained model')
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='Batchsize')
    parser.add_argument('-t', '--test_size', dest='test_size', help='Testset size')
    parser.add_argument('-v', '--validation_size', dest='validation_size', help='Validation set size')
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs to train the network')
    parser.add_argument('-n', '--net_name', dest='net_name', help='Choose CNN to train')
    parser.add_argument('-l', '--lr_step', dest='lr_step', help='True/False: Enable/Disable learning_rates scheduler')
    parser.add_argument('-s', '--lr_step_size', dest='lr_step_size', help='Decay learning_rate every lr_step_size')
    parser.add_argument('-g', '--lr_gamma', dest='lr_gamma', help='Decay learning_rate by lr_gamma')
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else '/Users/libao/Documents/data/animals/'
    model_dir = args.model_dir if args.model_dir else '../trained_model'
    batch_size = int(args.batch_size) if args.batch_size else 128
    test_size = float(args.test_size) if args.test_size else 0.25
    validation_size = float(args.validation_size) if args.validation_size else 0.2
    epochs = int(args.epochs) if args.epochs else 30
    lr_step = True if args.lr_step and args.lr_step == 'True' else False
    lr_step_size = int(args.lr_step_size) if args.lr_step_size else 10
    lr_gamma = float(args.lr_gamma) if args.lr_gamma else 0.1
    net_name = args.net_name if args.net_name else 'ShallowNet'

    # epochs = 5
    learning_rates = [1e-1, 1e-2]
    momentums = [0.85]
    # learning_rates = [1e-1, 1e-2,1e-3, 1e-4]
    # momentums = torch.arange(0.8, 1.00, 0.05)

    print('data_dir:', data_dir)
    print('model_dir:', model_dir)
    print('batch_size:', batch_size)
    print('test_size:', test_size)
    print('validation_size:', validation_size)
    print('epochs:', epochs)
    print('lr_step:', lr_step)
    print('lr_step_size:', lr_step_size)
    print('lr_gamma:', lr_gamma)
    print('learning_rates:', learning_rates)
    print('momentums:', momentums)
    print('net_name:', net_name)

    ct = CNNTrainer(data_dir, model_dir, batch_size, test_size,
                    validation_size, learning_rates, momentums, epochs,
                    net_name, lr_step, lr_step_size, lr_gamma)
    ct.test()