# 4. argparse를 활용한 Main 모듈 만들기

import torchvision.transforms as tr
from torch.utils.data import DataLoader
import resnet as RN
from dataset import cifar10
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train import training
from test import evaluation
from torchvision import transforms
import torch.nn.parallel

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def main():
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='인자값을 입력합니다.')

    # dataset, model, batch size, epoch, learning rate
    parser.add_argument('--model', '-m', required=False, default='resnet', help='Name of Model')
    parser.add_argument('--batch', '-b', required=False, default=32, help='Batch Size')
    parser.add_argument('--epoch', '-e', required=False, default=50, help='Epoch')
    parser.add_argument('--lr', '-l', required=False, default=0.001, help='Learning Rate')
    parser.add_argument('--depth', required=False, default=32, type=int, help='depth of the network (default: 32)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock for CIFAR datasets (default: bottleneck)')
    parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str, help='dataset cifar10')

    parser.set_defaults(bottleneck=True)

    best_err = 100

    # 입력받은 인자값을 args에 저장
    args = parser.parse_args()

    # 입력받은 인자값 출력
    print(args.model)
    print(args.batch)
    print(args.epoch)
    print(args.lr)
    print(args.depth)

    # model
    if args.model == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, 10, args.bottleneck)
    
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()
    print(model)

    # Data transforms (normalization & data augmentation)
    
    transf_train = tr.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])
    transf_test = ([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])

    trainset = cifar10(data_set_path="C:/Users/JHP/Desktop/cifar10/train", transforms=transf_train)
    testset = cifar10(data_set_path="C:/Users/JHP/Desktop/cifar10/test", transforms=transf_test)

    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3)

    for epoch in range(args.epoch):

        train_loss, train_accuracy = training(model, trainloader, criterion,  optimizer, scheduler)

        test_loss, test_accuracy = evaluation(model, testloader, criterion)

        print(f'Epoch:{epoch} Train loss:{train_loss} Train accuracy:{100*train_accuracy:.4f}%Test loss:{test_loss} Test accuracy:{100*test_accuracy:.4f}%')

        best_err = test_accuracy <= best_err
        best_err = min(test_accuracy, best_err)

        # 현재까지 학습한 것 중 best_err인 경우
        if best_err == test_accuracy:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoint.tar')



    model.load_state_dict(torch.load('model_weights.pth'))

if __name__ == "__main__":
    main()