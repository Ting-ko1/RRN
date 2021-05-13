import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import RRN
import csv
import codecs
import os

from torch.utils.data import DataLoader

eps=[0.5,0.75,1,1.25,1.5,2]
epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10           
random_seed = 1
torch.manual_seed(random_seed)
train_flag=False

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./MNIST', train=True, download=False,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
    batch_size=batch_size_train, shuffle=True)                                                      

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./MNIST', train=False, download=False,
                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 25, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.pool=nn.AdaptiveMaxPool2d(12)
        self.RRN=RRN.algorithm().cuda()
        
        self.fc1 = nn.Linear(625, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x,eps):
        x=self.RRN(x, eps).float()
        x=self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x=self.conv2(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(x), 2))

        x = x.view(-1, 625)                 
        x = F.relu(self.fc1(x.cuda()))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



train_losses = []
test_losses = []
accuracy=[]         #test accuracy
train_acc=[]



def train(network, optimizer,epoch, eps):
    network.train()         
    correct=0
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target=data.cuda(),target.cuda()
        optimizer.zero_grad()
        output = network(data, eps)
        loss = F.nll_loss(output, target)
        loss.backward()
        train_loss+=loss.item()
        optimizer.step()                
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    acc=correct.item()/len(train_loader.dataset);    
    train_acc.append(acc)
    print('train epoch: {} \tloss: {:.6f}\ttrain accuracy: {:.2f}%'.format(epoch,train_loss, 100. * correct.item()/len(train_loader.dataset))) 
    return train_loss, acc


def test(network, epoch, eps):
    network.eval()      
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data,target=data.cuda(),target.cuda()
            output = network(data, eps)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc=correct.item()/len(test_loader.dataset)
    accuracy.append(acc)
    print('\ntest set:  Avg. loss: {:.4f}, Accuracy:{}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))
    train_flag=True
    return test_loss, acc



def main():
    for eps_i in eps:
        network = Net()
        network.cuda()
        optimizer = optim.SGD(network.parameters(),
                       lr=learning_rate, momentum=momentum)
        dirname="../mnist/eps{0}".format(eps_i)
        checkDir(dirname)
        for epoch in range(1, epochs+1):
            train_loss, train_acc_ep= train(network, optimizer, epoch, eps_i)    
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
            #print("train_loss: {0}, train_acc: {1}, test_loss: {2}, test_acc: {3}".format(train_loss,train_acc_ep,test_loss,test_acc_ep))
            path_checkpoint=dirname+"/checkpoint_epoch_{0}.tar".format(epoch)
            torch.save(checkpoint,path_checkpoint)
    wrirteCsv("./result/mst_train_acc.csv", train_acc)          #results for each eps
    wrirteCsv("./result/mst_train_loss.csv", train_losses)            
    wrirteCsv("./result/mst_test_acc.csv", accuracy)
    wrirteCsv("./result/mst_test_loss.csv", test_losses)

def wrirteCsv(filename, datas):
    file_csv=codecs.open(filename, 'w', 'utf-8')
    write=csv.writer(file_csv, delimiter=' ')
    for data in datas:
        write.writerow(data)


def checkDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

main()
