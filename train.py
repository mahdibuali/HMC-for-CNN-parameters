import argparse
import logging
from   pyexpat import model
from   statistics import mode
import numpy as np
import torch
from   torchvision import datasets, transforms
import Model
import torchvision


def one_hot(y, n_classes):
    """Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot


def main(*ARGS):    
    #load data
    folder = "./data"

    batch_size = 64

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=folder, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=folder, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
    num_e = 50

    model = Model.Net()
    l = []
    for epochs in range(num_e):
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):

            input, label = data

            model.train_one_epoch(input, label, None, None)

            with torch.no_grad():

                predicted = model.predict(input)
                total += label.size(0)
                correct += (predicted == label).sum().item()
        l.append(100 * correct // total)
        print(100 * correct // total)
    print(l)
    print("done training")

    path = "MA_weights_64.ptnnp"

    torch.save(model.model.state_dict(), path)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            predicted = model.predict(images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
# [45, 60, 67, 72, 75, 77, 79, 81, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 94, 95, 95, 96, 96, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 99, 99, 99, 98, 99, 99, 99, 99]


if __name__ == '__main__':
    main()