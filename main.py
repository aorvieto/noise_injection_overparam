import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR100
import copy
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import flat_params, compute_jacobian
from models import CIFARCNN2, CIFARCNN3, MNISTNet1, MNISTNet2, MNISTNet3, CIFARCNN1, vgg11_bn, ResNet34
from datetime import datetime
from pyhessian import hessian
import torchvision.transforms as tt
from tqdm import tqdm


def train_net(GPU, settings):

    ########### Setting Up GPU ###########  
    gpu_ids = GPU
    torch.cuda.set_device(gpu_ids[0])
    device = 'cuda'

    ########### Setup Data and Model ###########    
    if settings["dataset"]=="MNIST":

        #data
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        validation_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=settings["bs"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=settings["bs"], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

        #model
        if settings["net"] == "MLP1": 
            model = torch.nn.DataParallel(MNISTNet1(),gpu_ids).cuda()
        elif settings["net"] == "MLP2":
            model = torch.nn.DataParallel(MNISTNet2(),gpu_ids).cuda()
        elif settings["net"] == "MLP3":
            model = torch.nn.DataParallel(MNISTNet3(),gpu_ids).cuda()
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss()

    elif settings["dataset"]=="FMNIST":

        #data
        train_dataset = datasets.FashionMNIST('./data',download=True, train= True, transform=transforms.ToTensor())
        validation_dataset = datasets.FashionMNIST('./data',download=True, train= False, transform=transforms.ToTensor())

        #Trainloader subset
        #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=settings["bs"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        subset = random.sample(range(train_dataset.data.shape[0]),settings["subset"])
        sample_ds = torch.utils.data.Subset(train_dataset, subset)
        sample_sampler = torch.utils.data.RandomSampler(sample_ds)
        train_loader = torch.utils.data.DataLoader(sample_ds, sampler=sample_sampler, batch_size=settings["bs"], num_workers=8, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=settings["bs"], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #hessloader subset
        hess_loader = train_loader

        #models
        if settings["net"] == "MLP1": 
            model = torch.nn.DataParallel(MNISTNet1(),gpu_ids).cuda()
        elif settings["net"] == "MLP2":
            model = torch.nn.DataParallel(MNISTNet2(),gpu_ids).cuda()
        elif settings["net"] == "MLP3":
            model = torch.nn.DataParallel(MNISTNet3(),gpu_ids).cuda()
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss()

    elif settings["dataset"]=="CIFAR":

        #data
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
        validation_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform)
        
        #trainloader subset
        #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=settings["bs"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        subset = random.sample(range(train_dataset.data.shape[0]),settings["subset"])
        sample_ds = torch.utils.data.Subset(train_dataset, subset)
        sample_sampler = torch.utils.data.RandomSampler(sample_ds)
        train_loader = torch.utils.data.DataLoader(sample_ds, sampler=sample_sampler, batch_size=settings["bs"], num_workers=8, pin_memory=True, persistent_workers=True)

        #hessloader subset
        subset = random.sample(range(train_dataset.data.shape[0]),int(settings["subset"]/10))
        sample_ds_hess = torch.utils.data.Subset(train_dataset, subset)
        sample_sampler_hess = torch.utils.data.RandomSampler(sample_ds_hess)
        hess_loader = torch.utils.data.DataLoader(sample_ds_hess, sampler=sample_sampler_hess, batch_size=settings["bs"], num_workers=8, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=settings["bs"], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #models
        if settings["net"] == "CNN1": 
            model = torch.nn.DataParallel(CIFARCNN1(),gpu_ids).cuda()
        elif settings["net"] == "CNN2": 
            model = torch.nn.DataParallel(CIFARCNN2(),gpu_ids).cuda()
        elif settings["net"] == "CNN3": 
            model = torch.nn.DataParallel(CIFARCNN3(),gpu_ids).cuda()
        elif settings["net"] == "CIFAR10Res34":
            model = torch.nn.DataParallel(ResNet34(),gpu_ids).cuda()

        else: print("model not defined")
        criterion = nn.CrossEntropyLoss() 

    elif settings["dataset"]=="CIFAR100":
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
        train_transform = tt.Compose([tt.RandomHorizontalFlip(),tt.RandomCrop(32,padding=4,padding_mode="reflect"),tt.ToTensor(), tt.Normalize(*stats)])
        test_transform = tt.Compose([tt.ToTensor(),tt.Normalize(*stats)])
        train_dataset = CIFAR100(download=True,root="./data",transform=train_transform)
        test_data = CIFAR100(root="./data",train=False,transform=test_transform)

        #trainloader subset
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=settings["bs"],num_workers=8,pin_memory=True,shuffle=True, persistent_workers=True)
       
        #hessloader
        subset = random.sample(range(train_dataset.data.shape[0]),int(settings["subset"]/25))
        sample_ds_hess = torch.utils.data.Subset(train_dataset, subset)
        sample_sampler_hess = torch.utils.data.RandomSampler(sample_ds_hess)    
        hess_loader = torch.utils.data.DataLoader(sample_ds_hess, sampler=sample_sampler_hess, batch_size=settings["bs"], num_workers=8, pin_memory=True, persistent_workers=True)
    
        #testloader       
        validation_loader = torch.utils.data.DataLoader(test_data,batch_size=settings["bs"],num_workers=8,pin_memory=True, persistent_workers=True)

        #models
        if settings["net"] == "CIFAR100vgg": 
            model = torch.nn.DataParallel(vgg11_bn(),gpu_ids).cuda()
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss() 

    # elif settings["dataset"]=="TinyImagenet":
    #     #todo


    ########### Setup Optimizer ###########   
    if settings["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=0.9)
    elif settings["optimizer"]=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])        
    else: print("method not defined!!")

    if settings["scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(settings["epochs"]))
        
    sigma_curr = settings["sigma"]

    ########### Setup Writer Variables ###########  
    results = {"rec_steps":[], "train_loss":[], "test_loss":[],"hess":[], "test_acc":[], "train_acc":[], "hess_trace":[], "l1_norm":[], "l2_norm":[], "grad_norm":[]}    

    ########### Getting number of layers ###########      
    n_groups = 0
    dim_model = 0
    with torch.no_grad():
        for param in model.parameters():   
            n_groups = n_groups + 1
            dim_model = dim_model + torch.numel(param)
    print('Model dimension: ' + str(dim_model))
    print('Number of groups: ' + str(n_groups))

    ##### iteration counter
    iter = 0
    
	########### Training ###########     
    for epoch in range(settings["epochs"]): 
        
        ########### Saving stats every few epochs ########### 
        if (epoch%settings["rec_step"])==0:
            results["rec_steps"].append(epoch)
            model.eval()
            
            #computing stats: train loss
            train_loss, correct = 0, 0
            for d in train_loader:
                data, target = d[0].to(device, non_blocking=True),d[1].to(device, non_blocking=True)
                output = model(data)
                train_loss += criterion(output, target).data.item()/len(train_loader)
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
            accuracy_train = 100. * correct.to(torch.float32) / len(train_loader.dataset)

            #computing stats: hessian
            hess_train = 0
            optimizer.zero_grad(set_to_none=True)
            for d in hess_loader:
                data, target = d[0].to(device, non_blocking=True),d[1].to(device, non_blocking=True)
                hess_train = hess_train+np.sum(hessian(model, criterion, data=(data, target), cuda=True).trace())/len(hess_loader) 
            optimizer.zero_grad(set_to_none=True)

            #computing stats: test loss
            test_loss, correct = 0, 0
            for data, target in validation_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                #d_out = output.size[-1]
                #print(d_out)
                test_loss += criterion(output, target).data.item()/len(validation_loader)
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
            accuracy_test = 100. * correct.to(torch.float32) / len(validation_loader.dataset)


            #regularized loss
            # J = torch.zeros((d_out, dim_model))
            # for data, target in validation_loader:
            #     data = data.to(device, non_blocking=True)
            #     target = target.to(device, non_blocking=True)
            #     output = model(data)
            #     J = J + torch.mean(compute_jacobian(data, output),0) / len(train_loader)            

            #saving stats
            results["train_loss"].append(train_loss)
            results["train_acc"].append(accuracy_train)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(accuracy_test)
            results["hess"].append(hess_train)
            results["l1_norm"].append(torch.norm(flat_params(model),1).cpu().detach().numpy())
            results["l2_norm"].append(torch.norm(flat_params(model),2).cpu().detach().numpy())
            #results["grad_norm"].append(grad_norm(model).numpy())

            print('Epoch {}: Train L: {:.4f}, TrainAcc: {:.0f}, Test L: {:.4f}, TestAcc: {:.0f} \n'.format(epoch, train_loss, accuracy_train, test_loss, accuracy_test))

        ########### Saving stats every few epochs ########### 
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            ### model perturbation
            if settings["noise"] != "no":
                param_copy = []
                with torch.no_grad():
                    i=0
                    for param in model.parameters():
                        param_copy.append(param.data)
                        if settings["noise"] == "all":
                            param.data = param.data + (sigma_curr/math.sqrt(n_groups))*torch.normal(0, 1, size=param.size(),device=device)
                        elif settings["noise"] == "layer":
                            if i==(iter%n_groups):
                                param.data = param.data + sigma_curr*torch.normal(0, 1, size=param.size(),device=device)
                            i = i+1

            ### backprop
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad() 
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            ### model recovery
            if settings["noise"] != "no":
                with torch.no_grad():
                    i=0
                    for param in model.parameters():
                        param.data = param_copy[i]
                        i=i+1

            optimizer.step()
            #print(loss.item())
            iter = iter +1

        if settings["scheduler"]:
            scheduler.step()

    return results

def settings_to_str(settings):
    return datetime.now().strftime("%H_%M_%S")+ '_' + settings["dataset"] + '_subset' + str(settings["subset"])  + "_" + settings["net"] + "_noise_" + settings["noise"] + '_bs' + str(settings["bs"]) + '_' + settings["optimizer"] + '_lr' + str(settings["lr"]) + '_sigma'+ str(settings["sigma"])+ '_rec'+ str(settings["rec_step"])

if __name__ == "__main__":

    if 1: 
        GPU = [6]
        dataset = "FMNIST"
        net = "MLP2"
        subset = 1024
        rec_step = 1000

        nep = 10000
        lr1 = 0.005
        bs = 1024

        noise = "no"

        sigma = 0.001
        settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":False, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        results = train_net(GPU,settings)
        torch.save(results, 'results/'+settings_to_str(settings)+'.pt')   

        # sigma = 0.005
        # settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":False, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        # results = train_net(GPU,settings)
        # torch.save(results, 'results/'+settings_to_str(settings)+'.pt')   

        # sigma = 0.05
        # settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":False, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        # results = train_net(GPU,settings)
        # torch.save(results, 'results/'+settings_to_str(settings)+'.pt')   

    if 0: 
        GPU = [7]
        dataset = "CIFAR"
        subset = 50000
        nep = 500
        lr1 = 0.01
        bs = 128
        rec_step = 25
        sigma = 0.05
        net = "CNN2"

        noise = "no" #layer,all
        sigma = 0.05
        lr1 = 0.005
        settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        results = train_net(GPU,settings)
        torch.save(results, 'results/'+settings_to_str(settings)+'.pt') 

        noise = "layer" #layer,all
        sigma = 0.05
        lr1 = 0.005
        settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        results = train_net(GPU,settings)
        torch.save(results, 'results/'+settings_to_str(settings)+'.pt')  

        noise = "all" #layer,all
        sigma = 0.05
        lr1 = 0.005
        settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        results = train_net(GPU,settings)
        torch.save(results, 'results/'+settings_to_str(settings)+'.pt')  


    if 0:
        GPU = [0,1]
        dataset = "CIFAR100"
        subset = 60000
        nep = 1000
        lr1 = 0.01
        bs = 128
        rec_step = 25
        sigma = 0.05
        net = "CIFAR100vgg"

        noise = "layer" #layer,all
        sigma = 0.5
        lr1 = 0.001
        settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"Adam", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        results = train_net(GPU,settings)
        torch.save(results, 'results/'+settings_to_str(settings)+'.pt')  

        # noise = "layer" #layer,all
        # sigma = 1
        # lr1 = 0.05
        # settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"Adam", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        # results = train_net(GPU,settings)


        # noise = "layer" #layer,all
        # sigma = 1
        # lr1 = 0.05
        # settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"Adam", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        # results = train_net(GPU,settings)
        # torch.save(results, 'results/'+settings_to_str(settings)+'.pt')  

    if 0:
        GPU = [0]
        dataset = "CIFAR"
        subset = 50000
        nep = 5000
        lr1 = 0.001
        bs = 1024
        rec_step = 1
        sigma = 0.05
        net = "CNN2"

        noise = "layer" #layer,all
        sigma = 0.05
        settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"Adam", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        results = train_net(GPU,settings)
        torch.save(results, 'results/'+settings_to_str(settings)+'.pt')  

        # noise = "all" #layer,all
        # sigma = 0.05
        # settings = {"dataset":dataset, "subset": subset, "net": net, "optimizer":"SGD", "scheduler":True, "noise":noise, "bs":bs, "lr":lr1, "sigma":sigma, "epochs":nep, "rec_step":rec_step}
        # results = train_net(GPU,settings)
        # torch.save(results, 'results/'+settings_to_str(settings)+'.pt')           

