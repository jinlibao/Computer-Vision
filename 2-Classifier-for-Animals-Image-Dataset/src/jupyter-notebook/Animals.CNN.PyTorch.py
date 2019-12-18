#!/usr/bin/env python
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

    def __init__(self):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        plt.style.use('ggplot')
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')

        root_dir = '/Users/libao/Documents/data/animals/'
        batch_size = 128
        dataset = self.read_image(root_dir, batch_size=batch_size)
        self.dataset = dataset
        train_loader, validation_loader, test_loader, classes = dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.classes = classes

    def imshow(self, image):
        image = image / 2 + 0.5

        # Method 1
        image_pil = transforms.ToPILImage()(image)
        plt.imshow(image_pil)

        # # Method 2
        # image_np = image.numpy()
        # plt.imshow(np.transpose(image_np, (1, 2, 0)))

        plt.show()

    def plot(self, df, net_name, learning_rates, momentums):
        for lr in learning_rates:
            for m in momentums:
                d = df[(df['lr'] == lr) & (df['momentum'] == m)]
                fig = plt.figure()
                plt.plot(d['epoch'], d['acc_train'], label='acc_train')
                plt.plot(d['epoch'], d['acc_val'], label='acc_val')
                plt.plot(d['epoch'], d['loss_train'], label='loss_train')
                plt.plot(d['epoch'], d['loss_val'], label='loss_val')
                plt.xlabel('epoch #')
                plt.ylabel('loss/accuracy')
                plt.title(net_name)
                plt.legend()
                plt.show()
                plt.close()

    def read_image(self, root_dir, test_size=0.25, validation_size=0.2, batch_size=32, transform=None):
        # Read images and labels
        data, labels = [], []
        classes = [c for c in os.listdir(root_dir) if not c.startswith('.')]
        paths = [os.path.join(root_dir, c) for c in classes]
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
        if not transform:
            transform = transforms.Compose([
                transforms.Resize((32, 32)), # resize all the image to 32x32x3
                transforms.ToTensor(),       # rescale images from [0, 255] to [0, 1.0], (H x W x C) to (C x H x W)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # rescale [0, 1.0] to [-1.0, 1.0]
            ])

        validation_set = AnimalsDataset(validation_set, transform)
        train_set = AnimalsDataset(train_set, transform)
        test_set = AnimalsDataset(test_set, transform)

        # Create DataLoader for train_set, validation_set, and test_set
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        return (train_loader, validation_loader, test_loader, classes)

    def get_report(self, net, data_loader):
        with torch.no_grad():
            predicts_total, targets_total = [], []
            for batch in data_loader:
                images, targets = batch
                outputs = net(images)
                _, predicts = torch.max(outputs, 1)
                predicts_total.extend(predicts.numpy())
                targets_total.extend(targets.numpy())
            confusion = confusion_matrix(targets_total, predicts_total)
            report_dict = classification_report(targets_total, predicts_total, target_names=self.classes, output_dict=True)
            report = classification_report(targets_total, predicts_total, target_names=self.classes)
        return (report_dict, report, confusion)

    def model(self, net, data_loader, criterion=None, optimizer=None, mode='eval'):
        running_loss = 0.0
        for batch in data_loader:
            images, targets = batch
            if mode == 'train':
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            else:
                outputs = net(images)
                loss = criterion(outputs,targets)
            running_loss += loss.item() / len(data_loader)
        report_dict, _, _ = self.get_report(net, data_loader)

        return (running_loss, report_dict['accuracy'])

    def CNN(self, net, dataset, epochs=50, lr=1e-2, momentum=8.5e-1, dest='../../trained_model'):
        train_loader, validation_loader, test_loader, classes = dataset
        batch_size = train_loader.batch_size
        # Instantiate CNN, pick loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) # lr: 1e-1, 1e-2, 1e-3, 1e-4; momentum: 0.9-0.99

        history = []
        nets = []
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        for epoch in range(epochs):
            loss_train, acc_train = self.model(net, train_loader, criterion, optimizer, 'train')
            lr_scheduler.step()
            loss_val, acc_val = self.model(net, validation_loader, criterion)
            nets.append(net.state_dict())
            history.append({
                'epoch': epoch + 1,
                'acc_train': acc_train,
                'acc_val': acc_val,
                'loss_train': loss_train,
                'loss_val': loss_val,
                'lr': lr,
                'momentum': momentum
            })
            print('[epoch {:2d}/{:2d}] loss_train: {:5.3f}, acc_train: {:5.3f}, loss_val: {:5.3f}, acc_val: {:5.3f}'.format(
                epoch + 1, epochs, loss_train, acc_train, loss_val, acc_val))
        print('Finished training.')

        idx = np.argmax(np.array(pd.DataFrame(history)['acc_val']))
        print('The best epoch is {:d}'.format(idx + 1))

        if not os.path.exists(dest):
            os.mkdir(dest)
        model_filename = '{:s}/{:s}_{:.1e}_{:.1e}_{:d}.pth'.format(dest, net.name, lr, momentum, epochs)
        print('Saving trained model to {:s}'.format(model_filename))
        torch.save(nets[idx], model_filename)

        print('Evaluating network at best epoch...')
        net.load_state_dict(nets[idx])
        # Calculate the accuracy and generate classification report
        report_dict, report, confusion = self.get_report(net, test_loader)

        return (report_dict, report, confusion, history)


    def test(self):
        train_loader, validation_loader, test_loader, classes = self.dataset
        print(len(train_loader) * train_loader.batch_size)
        print(len(test_loader) * test_loader.batch_size)
        print(len(validation_loader) * validation_loader.batch_size)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        epochs = 3
        # learning_rates = [1e-1, 1e-2,1e-3, 1e-4]
        # momentums = torch.arange(0.8, 1.00, 0.05)
        learning_rates = [1e-2]
        momentums = [8.5e-1]

        nets = [
            MiniVGGNet(),
            LeNet(),
            ShallowNet(),
            TorchNet()
        ]

        for net in nets:
            print('Training {:s}...'.format(net.name))
            all_history = []
            for lr in learning_rates:
                for momentum in momentums:
                    dest='../../trained_model/{:s}'.format(net.name)
                    print('learning rate: {:e}, momentumn: {:e}'.format(lr, momentum))
                    report_dict, report, confusion, history = self.CNN(net, self.dataset, epochs, lr, momentum, dest)
                    pd.DataFrame(report_dict).T.to_csv('{:s}/{:s}_{:.1e}_{:.1e}_{:d}_report.csv'.format(dest, net.name, lr, momentum, epochs))
                    all_history.extend(history)
                    print('Confusion matrix:\n', confusion)
                    print('Classification report:\n', report)
            df = pd.DataFrame(all_history)
            df.to_csv('{:s}/training_history.csv'.format(dest))
            self.plot(df, net.name, learning_rates, momentums)


if __name__ == '__main__':
    ct = CNNTrainer()
    ct.test()
