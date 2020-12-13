import torch
import torch.nn as nn
from model import CNNmodel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import os

def weights_init(m):
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # Apply weight initial by finding the class     #
    # name of layer.                                #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def mytest(opt, model, device ,criterion, test_loader,f):
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # testing the data in dataloader                #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)/opt.batchSize
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),file = f)

def mytrain(opt,model,device,train_loader,optimizer,critetion,epoch,f):
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # training the data from data laoder            #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = critetion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()),file = f)
            print('loss: {:.6f}'.format(loss.item()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datat',default = "D:\\DL\\DL_HW2\\animal-10\\train",help='path to dataset')
    parser.add_argument('--datav',default = "D:\\DL\\DL_HW2\\animal-10\\val",help='path to val')
    parser.add_argument('--workers',default=4, type=int, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--step', type=int, default=301, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manualSeed',default = 87, type=int, help='manual seed')
    parser.add_argument('--saveroot',default = './result/',help = 'save path')
    opt = parser.parse_args()

    try:
        os.mkdir(opt.saveroot)
    except OSError:
        pass

    kwargs = {'num_workers': opt.workers, 'pin_memory': True} if torch.cuda.is_available and opt.cuda else {}
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # Read the dataset by pytorch API               #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    dataset = dset.ImageFolder(root=opt.datat,
                               transform = transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize( (0.5077,0.4810,0.3990),(0.2638,0.2592,0.2717) )])
                               )
    vdataset = dset.ImageFolder(root=opt.datav,
                               transform = transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize( (0.5077,0.4810,0.3990),(0.2638,0.2592,0.2717) )])
                               )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True,**kwargs)
    vdataloader = torch.utils.data.DataLoader(vdataset,batch_size =opt.batchSize,shuffle = True, **kwargs)

    device = torch.device("cuda:0" if opt.cuda and torch.cuda.is_available else "cpu")
    model = CNNmodel(3,10).to(device)
    model.apply(weights_init)
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # Restore from checkpoint, if any.              #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    
    #model.load_state_dict(torch.load('result/45modelv3.pt'))
    optim_ = optim.Adam(model.parameters(),lr = opt.lr,betas=(opt.beta1,0.999))
    crite = nn.CrossEntropyLoss()
    
    f = open(opt.saveroot + 'log.csv','w+')
    f2 = open(opt.saveroot + 'acclog.csv','w+')
    for epoch in range(46,opt.step):
        mytrain(opt,model,device,dataloader,optim_,crite,epoch,f)
        print('train:',file = f2)
        mytest(opt,model,device,crite,dataloader,f2)
        print('val:',file = f2)
        mytest(opt,model,device,crite,vdataloader,f2)
        if epoch % 5 == 0:
            torch.save(model.state_dict(),opt.saveroot+'{}modelv3.pt'.format(epoch))

if __name__ == "__main__":
    main()