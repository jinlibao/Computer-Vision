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


class AnimalsDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        image = self.dataset[0][idx]
        target = self.dataset[1][idx]
        if self.transform:
            image = self.transform(image)
        return [image, target]


class TransferLearner(object):

    def __init__(self, data_dir, model_dir, batch_size, test_size,
                 validation_size, learning_rate, momentum, epochs,
                 net_name, lr_step, lr_step_size, lr_gamma, weight_decay,
                 use_data_augmentation, use_dropout, use_batch_norm):
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
        self.use_data_augmentation = 1 if use_data_augmentation else 0
        self.test_transform = transforms.Compose([
                transforms.Resize((32, 32)),                            # resize all the image to 32x32x3
                transforms.ToTensor(),                                  # [0, 255] -> [0, 1.0], (H x W x C) -> (C x H x W)
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        ])
        if use_data_augmentation:  # Data Augmentation
            self.train_transform = transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((32, 32)),                            # resize all the image to 32x32x3
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.train_transform = self.test_transform

        self.dataset = self.read_image(data_dir, test_size, validation_size,
                                       batch_size)
        train_loader, validation_loader, test_loader, classes = self.dataset
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.classes = classes

        # Tuning hyperparamters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.lr_step = lr_step
        self.use_lr_step = 1 if lr_step else 0
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        # Choose device to train on
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        # Select CNN
        self.net_name = net_name
        self.use_dropout = 1 if use_dropout else 0
        self.use_batch_norm = 1 if use_batch_norm else 0

    def imshow(self, image):
        image = image / 2 + 0.5

        # Method 1
        image_pil = transforms.ToPILImage()(image)
        plt.imshow(image_pil)

        # # Method 2
        # image_np = image.numpy()
        # plt.imshow(np.transpose(image_np, (1, 2, 0)))

        plt.show()

    def plot(self, df, net_name, learning_rate, momentum, weight_decay,
             lr_gamma, plot_dir='.', suffix='', title=''):
        linestyle = self.linestyle
        d = df[(df['weight_decay'] == weight_decay) & (df['lr_gamma'] == lr_gamma)]
        plt.figure(figsize=(8, 6))
        plt.plot(d['epoch'], d['acc_train'], linestyle[0], label='acc_train')
        plt.plot(d['epoch'], d['acc_val'], linestyle[1], label='acc_val')
        plt.plot(d['epoch'], d['loss_train'], linestyle[2], label='loss_train')
        plt.plot(d['epoch'], d['loss_val'], linestyle[3], label='loss_val')
        plt.xlabel('epoch #', color='black')
        plt.ylabel('loss/accuracy', color='black')
        if len(title) > 0:
            plt.title(title)
        plt.legend(loc='best')
        [i.set_color('black') for i in plt.gca().get_xticklabels()]
        [i.set_color('black') for i in plt.gca().get_yticklabels()]
        plt.show(block=False)
        filename = '{:s}/{:s}_{:s}.pdf'.format(
            plot_dir, net_name, suffix
        )
        print('Saving plot to {:s}'.format(filename))
        plt.savefig(filename, bbox_inches='tight')

    def read_image(self, data_dir, test_size=0.25, validation_size=0.2,
                   batch_size=32):
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
        validation_set = AnimalsDataset(validation_set, self.test_transform)
        train_set = AnimalsDataset(train_set, self.train_transform)
        test_set = AnimalsDataset(test_set, self.test_transform)

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
                net.train()
                optimizer.zero_grad()
                outputs = net(images)
                outputs = outputs.to(self.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            else:
                net.eval()
                outputs = net(images)
                outputs = outputs.to(self.device)
                loss = criterion(outputs, targets)
            running_loss += loss.item() / len(data_loader)
        report_dict, _, _ = self.get_report(net, data_loader)

        return (running_loss, report_dict['accuracy'])

    def get_pretrained_model(self, net_name='ResNet18', learning_rate=0.001,
                             momentum=0.9, weight_decay=0, lr_step_size=7,
                             lr_gamma=0.1, num_classes=3):
        if net_name == 'ResNet18':
            net = torchvision.models.resnet18(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        elif net_name == 'ResNet34':
            net = torchvision.models.resnet34(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        elif net_name == 'ResNet50':
            net = torchvision.models.resnet50(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        elif net_name == 'ResNet101':
            net = torchvision.models.resnet101(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        elif net_name == 'ResNet152':
            net = torchvision.models.resnet152(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        elif net_name == 'AlexNet':
            net = torchvision.models.alexnet(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(num_ftrs, num_classes)
            parameters = net.classifier[6].parameters()
        elif net_name == 'SqueezeNet':
            net = torchvision.models.squeezenet1_1(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1),
                                          stride=(1, 1))
            parameters = net.classifier[1].parameters()
        elif net_name == 'DenseNet':
            net = torchvision.models.densenet121(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.classifier.in_features
            net.classifier = nn.Linear(num_ftrs, num_classes)
            parameters = net.classifier.parameters()
        elif net_name == 'GoogLeNet':
            net = torchvision.models.googlenet(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        elif net_name == 'MnasNet':
            net = torchvision.models.mnasnet1_0(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(num_ftrs, num_classes)
            parameters = net.classifier[1].parameters()
        elif net_name == 'VGG11':
            net = torchvision.models.vgg11(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(num_ftrs, num_classes)
            parameters = net.classifier[6].parameters()
        elif net_name == 'VGG19':
            net = torchvision.models.vgg19(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(num_ftrs, num_classes)
            parameters = net.classifier[6].parameters()
        elif net_name == 'ShuffleNet':
            net = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()
        else:
            net = torchvision.models.resnet18(pretrained=True)
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, num_classes)
            parameters = net.fc.parameters()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(parameters,
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=lr_step_size,
                                                 gamma=lr_gamma)
        return (net, criterion, optimizer, lr_scheduler)


    def CNN_Transfer(self, net_name, dataset, epochs, lr, momentum, dest, lr_step,
            lr_step_size, lr_gamma, weight_decay, suffix):
        train_loader, validation_loader, test_loader, classes = dataset
        # Instantiate CNN, pick loss function and optimizer
        net, criterion, optimizer, lr_scheduler = self.get_pretrained_model(
            net_name, learning_rate, momentum, weight_decay, lr_step_size,
            lr_gamma)
        history = []
        nets = []
        for epoch in range(epochs):
            loss_train, acc_train = self.model(net, train_loader, criterion, optimizer, 'train')
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
                'momentum': momentum,
                'lr_gamma': lr_gamma,
                'weight_decay': weight_decay,
                'use_lr_step': self.use_lr_step,
                'use_data_augmentation': self.use_data_augmentation,
                'use_dropout': self.use_dropout,
                'use_batch_norm': self.use_batch_norm
            })
            print('epoch [{:d}/{:d}] loss_train: {:5.3f}, acc_train: {:5.3f}, loss_val: {:5.3f}, acc_val: {:5.3f}'.format(
                epoch + 1, epochs, loss_train, acc_train, loss_val, acc_val))
        print('Finished training.')

        idx = np.argmax(np.array(pd.DataFrame(history)['acc_val']))
        print('The best epoch is {:d}'.format(idx + 1))

        if not os.path.exists(dest):
            os.mkdir(dest)
        model_filename = '{:s}/{:s}_{:s}.pth'.format(
            dest, net_name, suffix
        )
        print('Saving trained model to {:s}'.format(model_filename))
        torch.save(nets[idx], model_filename)

        print('Evaluating network at best epoch...')
        net.load_state_dict(nets[idx])
        # Calculate the accuracy and generate classification report
        report_dict, report, confusion = self.get_report(net, test_loader)

        return (report_dict, report, confusion, history)

    def test_transfer_learning(self):
        train_loader, validation_loader, test_loader, classes = self.dataset
        net_name = self.net_name
        epochs = self.epochs
        learning_rate = self.learning_rate
        momentum = self.momentum
        dataset = self.dataset
        model_dir = self.model_dir
        lr_step = self.lr_step
        lr_step_size = self.lr_step_size
        lr_gamma = self.lr_gamma
        weight_decay = self.weight_decay
        suffix = '{:.2e}_{:.2e}_{:.2e}_{:.2e}_{:d}_{:d}_{:d}_{:d}_{:d}'.format(
            learning_rate, momentum, lr_gamma, weight_decay, self.use_lr_step,
            self.use_data_augmentation, self.use_dropout, self.use_batch_norm,
            epochs
        )
        dest = '{:s}/{:s}'.format(model_dir, net_name)
        if not os.path.exists(dest):
            os.makedirs(dest)

        if self.use_cuda:
            print('Training {:s} on GPU...'.format(net_name))
        else:
            print('Training {:s}...'.format(net_name))
        print('learning rate: {:e}, momentumn: {:e}, weight_decay: {:e} lr_gamma: {:e}'.format(
            learning_rate, momentum, weight_decay, lr_gamma
        ))
        report_dict, report, confusion, history = self.CNN_Transfer(
            net_name,  dataset, epochs, learning_rate, momentum, dest, lr_step,
            lr_step_size, lr_gamma, weight_decay, suffix)
        pd.DataFrame(report_dict).T.to_csv('{:s}/{:s}_report_{:s}.csv'.format(
            dest, net_name, suffix))
        print('Confusion matrix:\n', confusion)
        print('Classification report:\n', report)
        df = pd.DataFrame(history)
        df.to_csv('{:s}/{:s}_training_history_{:s}.csv'.format(
            dest, net_name, suffix
        ))
        title = '{:s}: DA: ${:d}$, Dropout: ${:d}$, BN: ${:d}$'.format(
            net_name, self.use_data_augmentation, self.use_dropout,
            self.use_batch_norm)
        # title = '{:s}: lr: ${:.4f}$, m: ${:.2f}$, wd: ${:.4f}$ $\gamma$: ${:.4f}$'.format(
        #     net_name, learning_rate, momentum, weight_decay, lr_gamma)
        self.plot(df, net_name, learning_rate, momentum, weight_decay, lr_gamma,
                  dest, suffix, title)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', help='Root directory of animals dataset')
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='Folder to store the trained model')
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='Batchsize')
    parser.add_argument('-t', '--test_size', dest='test_size', help='Testset size')
    parser.add_argument('-v', '--validation_size', dest='validation_size', help='Validation set size')
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs to train the network')
    parser.add_argument('-n', '--net_name', dest='net_name', help='Choose CNN to train')
    parser.add_argument('-r', '--learning_rate', dest='learning_rate', help='Learning rate')
    parser.add_argument('-w', '--weight_decay', dest='weight_decay', help='Weight decay')
    parser.add_argument('-u', '--momentum', dest='momentum', help='Momentum')
    parser.add_argument('-l', '--lr_step', dest='lr_step', help='True/False: Enable/Disable learning_rates scheduler')
    parser.add_argument('-s', '--lr_step_size', dest='lr_step_size', help='Decay learning_rate every lr_step_size')
    parser.add_argument('-g', '--lr_gamma', dest='lr_gamma', help='Decay learning_rate by lr_gamma')
    parser.add_argument('-a', '--use_data_augmentation', dest='use_data_augmentation', help='Use data_augmentation')
    parser.add_argument('-o', '--use_dropout', dest='use_dropout', help='Use dropout')
    parser.add_argument('-c', '--use_batch_norm', dest='use_batch_norm', help='Use batch normalization')
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else '/Users/libao/Documents/data/cnn/animals/'
    model_dir = args.model_dir if args.model_dir else '../trained_model'
    batch_size = int(args.batch_size) if args.batch_size else 128
    test_size = float(args.test_size) if args.test_size else 0.25
    validation_size = float(args.validation_size) if args.validation_size else 0.2
    epochs = int(args.epochs) if args.epochs else 20
    learning_rate = float(args.learning_rate) if args.learning_rate else 1e-3
    weight_decay = float(args.weight_decay) if args.weight_decay else 0
    momentum = float(args.momentum) if args.momentum else 0.9
    lr_step = True if args.lr_step and args.lr_step == 'True' else False
    lr_step_size = int(args.lr_step_size) if args.lr_step_size else 50
    lr_gamma = float(args.lr_gamma) if args.lr_gamma else 0.1
    net_name = args.net_name if args.net_name else 'ResNet18'
    use_data_augmentation = True if args.use_data_augmentation and args.use_data_augmentation == 'True' else False
    use_dropout = True if args.use_dropout and args.use_dropout == 'True' else False
    use_batch_norm = True if args.use_batch_norm and args.use_batch_norm == 'True' else False

    print('---------- Parameters ---------')
    print('data_dir:', data_dir)
    print('model_dir:', model_dir)
    print('batch_size:', batch_size)
    print('test_size:', test_size)
    print('validation_size:', validation_size)
    print('epochs:', epochs)
    print('lr_step:', lr_step)
    print('lr_step_size:', lr_step_size)
    print('lr_gamma:', lr_gamma)
    print('learning_rate:', learning_rate)
    print('weight_decay:', weight_decay)
    print('momentum:', momentum)
    print('net_name:', net_name)
    print('use_data_augmentation:', use_data_augmentation)
    print('use_dropout:', use_dropout)
    print('use_batch_norm:', use_batch_norm)

    tl = TransferLearner(data_dir, model_dir, batch_size, test_size,
                    validation_size, learning_rate, momentum, epochs,
                    net_name, lr_step, lr_step_size, lr_gamma, weight_decay,
                    use_data_augmentation, use_dropout, use_batch_norm)
    tl.test_transfer_learning()
