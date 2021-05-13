import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import RRN
import csv
import codecs
import time
import os

from torch.utils.data import DataLoader

eps = [0.5,0.75,1,1.5,2]
epochs = 300
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10  
random_seed = 1
torch.manual_seed(random_seed)
train_flag = False

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./CIFAR', train=True, download=False,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                                           torchvision.transforms.RandomGrayscale(), torchvision.transforms.ToTensor()])),
    batch_size=batch_size_train, shuffle=True)  

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./CIFAR', train=False, download=False,
                                 transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
    batch_size=batch_size_test, shuffle=True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.RRN = RRN.algorithm().cuda()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        
        self.fc1 = nn.Linear(512*4*4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x, eps):
        x = self.RRN(x, eps).float()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = F.relu(self.bn1(x))

        

        x = self.conv3(x)
        x = self.conv4(x)
        x = F.dropout2d(x)
        x = self.pool2(x)
        x = F.relu(self.bn2(x))
        

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = F.relu(self.bn3(x))
        

        x = self.conv8(x)
        x = F.dropout2d(x)
        x = self.conv9(x)
        x = self.conv10(x)  
        x = self.pool4(x)
        x = F.relu(self.bn4(x))
        
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.dropout2d(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = F.relu(self.bn5(x))
        

        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc1(x.cuda()))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return F.log_softmax(x)



train_losses = []
test_losses = []
accuracy = []
train_acc=[]



def train(network,optimizer,epoch, eps):
    network.train()  
    correct=0
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = network(data, eps)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss+=loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        optimizer.step()  
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    acc= correct.item()/len(train_loader.dataset)    
    train_acc.append(acc)
    print('train epoch: {}\t train accuracy: {:.2f}%\tloss: {:.6f}'.format(epoch, 100. * correct.item()/len(train_loader.dataset),train_loss))
    return train_loss, acc


def test(network,epoch, eps):
    network.eval()  
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = network(data, eps)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc= correct.item()/len(test_loader.dataset)
    accuracy.append(acc)
    print('\ntest set:  Avg. loss: {:.4f}, Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))
    return test_loss, acc


def wrirteCsv(filename, datas):
    file_csv = codecs.open(filename, 'w', 'utf-8')
    write = csv.writer(file_csv, delimiter=' ')
    for data in datas:
        write.writerow(data)


def main():
    
    for eps_i in eps:
        network = Net()
        network.cuda()
        optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate, momentum=momentum, weight_decay=1e-3)            
        dirname="../cifar/eps{0}".format(eps_i)
        checkDir(dirname)
        for epoch in range(1, epochs+1):
            train_loss, train_acc_ep= train(network,optimizer,epoch, eps_i)
            test_loss, test_acc_ep= test(network,epoch, eps_i)
            checkpoint={
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc_ep,
                "test_loss": test_loss,
                "test_acc": test_acc_ep
            }
            print("train_loss: {0}, train_acc: {1}, test_loss: {2}, test_acc: {3}".format(train_loss,train_acc_ep,test_loss,test_acc_ep))
            path_checkpoint=dirname+"/checkpoint_epoch_{0}.tar".format(epoch)
            torch.save(checkpoint,path_checkpoint)
    wrirteCsv('./result/cifar_train_loss.csv', train_losses)
    wrirteCsv('./result/cifar_train_acc.csv', train_acc)
    wrirteCsv('./result/cifar_test_loss.csv', test_losses)
    wrirteCsv('./result/cifar_test_acc.csv', accuracy)


def checkDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

main()

