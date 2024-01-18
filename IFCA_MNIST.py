import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scipy.special import binom

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init

from collections import Counter
import operator
import copy
from itertools import product,combinations
from time import time
#from IPython.core.display import display

#%matplotlib inline

## code extracted from https://www.kaggle.com/code/graymant/breast-cancer-diagnosis-with-pytorch
## SV code extracted from https://github.com/mburaksayici/ExplainableAI-Pure-Numpy/blob/main/KernelSHAP-Pure-Numpy.ipynb

import yaml
import os
import shutil

# Combinatorial UCB
import math
from models import Net 
from data import FEMNIST_dataset, Partition_MNIST_NIID
from data import MNIST_NIID_dataset 
import tensorflow.keras as tk 
import random 

import collections
from time import time
import random

def init():
    with open('settings/train_settings_bandits.yaml', 'r') as file:
        settings = yaml.safe_load(file)
    if not os.path.isdir('checkpoints_bandits'):
        os.mkdir('checkpoints_bandits')
    if not os.path.isdir(os.path.join('checkpoints_bandits', settings['experiment_name'])):
        os.mkdir(os.path.join('checkpoints_bandits', settings['experiment_name']))
    save_dir = os.path.join('checkpoints_bandits', settings['experiment_name'])
    if not os.path.isdir(os.path.join(save_dir, 'model')):
        os.mkdir(os.path.join(save_dir, 'model'))
    shutil.copyfile('settings/train_settings_bandits.yaml', save_dir + '/train_settings.yaml')
    return settings,save_dir

class IFCA():
    def __init__(self ,total_clients ,train_data ,train_partition ,val_partition ,test_data ,test_partition,n_clients_selected
                 ,alpha = 0.25, test='AFPL'):
        self.network = Net('MNIST_niid')
        self.total_clients = total_clients
        self.client_models = {}
        self.optimizers = {}
        self.dataloaders = {}
        self.len = {}
        self.len_test = {}
        self.len_really_test = {}
        self.dataloaders_test = {}
        self.dataloaders_really_test = {}
        self.best_test_loss = {}
        self.best_test_loss_global = 1000000
        self.current_test_loss = {}
        self.current_train_loss = {}
        self.test = test
       

        for i in range(total_clients):
            self.client_models[str(i)] = copy.deepcopy(self.network).double().cuda()
            self.optimizers[str(i)] = torch.optim.SGD(self.client_models[str(i)].parameters() ,lr=0.008 ,momentum=0.5)
            if data_fraction != 1:
                dataset_train= MNIST_NIID_dataset(train_data[0][blub] ,train_data[1][blub] ,train_partition ,i)
            else:
                dataset_train = MNIST_NIID_dataset(train_data[0], train_data[1], train_partition, i)

            if i == 1:
                if data_fraction != 1:
                    dataset = torch.utils.data.ConcatDataset([dataset_train,MNIST_NIID_dataset(train_data[0][blub] ,train_data[1][blub] ,train_partition,i )])
                else:
                    dataset = torch.utils.data.ConcatDataset(
                        [dataset_train, MNIST_NIID_dataset(train_data[0], train_data[1], train_partition, i)])
            if i > 1 :
                if data_fraction != 1:
                    dataset = torch.utils.data.ConcatDataset(
                    [dataset,MNIST_NIID_dataset(train_data[0][blub] ,train_data[1][blub] ,train_partition,i )])
                else:
                    dataset = torch.utils.data.ConcatDataset(
                        [dataset, MNIST_NIID_dataset(train_data[0], train_data[1], train_partition, i)])


            self.len[str(i) ]= len(dataset_train)
            self.dataloaders[str(i)] = DataLoader(dataset_train ,batch_size=16 ,shuffle=True)
            if data_fraction !=1:
                dataset_test= MNIST_NIID_dataset(train_data[0][blub] ,train_data[1][blub] ,val_partition,i  )
            else:
                dataset_test = MNIST_NIID_dataset(train_data[0], train_data[1], val_partition, i)

            dataset_really_test = MNIST_NIID_dataset(test_data[0],test_data[1],test_partition,i)
            self.len_really_test[str(i)] = len(dataset_really_test)
            self.dataloaders_really_test[str(i)] = DataLoader(dataset_really_test,batch_size=16,shuffle=True)
            self.len_test[str(i)] = len(dataset_test)
            self.dataloaders_test[str(i)] = DataLoader(dataset_test ,batch_size=16 ,shuffle=False)
            self.best_test_loss[str(i)] = 10000000
            self.current_test_loss[str(i)] = 100000
            self.current_train_loss[str(i)] = 1000000
          
        self.dataset_train = dataset_train
        self.dataloader_centralized = DataLoader(dataset, batch_size=32, shuffle=True)
        
    def init_ifca(self, k): 
        # setup models 
        self.global_models_ifca = {}
        for i in range(k):
            self.global_models_ifca[str(i)] = copy.deepcopy(self.network).double().cuda()

        # initialize cluster assignment to be random 
        self.cluster_assign = np.random.randint(0, k, size=self.total_clients)
    
    def run_ifca(self, selected_clients): 
        
        for idx, i in enumerate(selected_clients): 
            # find current cluster assignment 
            k_i = self.cluster_assign[i]
            # extract the appropriate model 
            model = copy.deepcopy(self.global_models_ifca[str(k_i)])
            dataloader = self.dataloaders[str(i)]
            optimizer = torch.optim.Adam(model.parameters() ,lr=0.001 *0.95**self.iteration)
            # perform local training 
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target =target.long().cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output ,target)
                loss.backward()
                optimizer.step()
            
            # save the trained model at the client side 
            self.client_models[str(i)] = copy.deepcopy(model)
            
    def cluster_ifca(self, selected_clients, k): 
        
        losses = np.zeros((len(selected_clients),k))
        for idx, i in enumerate(selected_clients): 
            dataloader = self.dataloaders[str(i)]
            for batch_idx, (data, target) in enumerate(dataloader):
                for k_i in range(k): 
                    data = data.double().cuda()
                    target =target.long().cuda()
                    output = self.global_models_ifca[str(k_i)](data)
                    loss = F.nll_loss(output ,target)
                    losses[i,k_i] += loss # or idx? 
                    
        #print(losses)
        print(np.min(losses,axis=1))
        self.cluster_assign = np.argmin(losses,axis=1)
        
    def combine_ifca(self, selected_clients, k):
        print(self.cluster_assign)
        for k_i in range(k): 
            clients_in_ki = [i for i in selected_clients if self.cluster_assign[i] == k_i]
            
            if len(clients_in_ki) > 0: 
                # do this only if there's at least once client per thing 
                shared_model = copy.deepcopy(self.global_models_ifca[str(k_i)]).double().cuda()
                n_clients = len(clients_in_ki)
                weight = [1/n_clients for x in range(n_clients)]

                for idx, i in enumerate(clients_in_ki):
                    for (name, param), (name2, param2) in zip(shared_model.named_parameters()
                            , self.client_models[str(i)].named_parameters()):
                        if idx == 0:
                            param.data = torch.zeros(param.shape).cuda().double()
                        param.data += weight[idx] * param2.data

                self.global_models_ifca[str(k_i)] = shared_model.double().eval()

    def update_local_models(self ,selected_clients):
        self.dw = {}
        loss_test = 0
        loss_test2 = 0
        losses = 0
        losses2 = 0
        loss_test3 = 0
        losses3 = 0

        for idx ,i in enumerate(selected_clients):
            self.client_models[str(i)].eval()
            dataloader_test = self.dataloaders_test[str(i)]
            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.double().cuda()
                    target =target.long().cuda()

                    output = self.client_models[str(i)](data)
                    loss_test += F.nll_loss(output ,target)
                self.current_test_loss[str(i)] = loss_test /self.len_test[str(i)]
                if self.current_test_loss[str(i)] < self.best_test_loss[str(i)]:
                    torch.save(self.client_models[str(i)].state_dict(), os.path.join(save_dir, 'model', 'best_model ' +str(i ) +'.pt'))
                    self.best_test_loss[str(i)] = self.current_test_loss[str(i)]

            losses += loss_test /self.len_test[str(i)]
            loss_test2 = 0
            self.client_models[str(i)].eval()
            dataloader = self.dataloaders[str(i)]
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = self.client_models[str(i)](data)
                    loss_test2 += F.nll_loss(output, target)

            losses2 += loss_test2 / self.len[str(i)]
            self.current_train_loss[str(i)] = loss_test2 / self.len[str(i)]

        print('full train loss: ', losses2)
        print('full loss: ', losses)

        return losses2, losses


    
    def calc_accuracy(self, dataloader, length):
        accuracies = np.zeros(len(self.selected_clients))
        total = 0
        self.accuracy_list = []
        for i in self.selected_clients:
            intermediate_accuracy = 0

            for batch_idx, (data, target) in enumerate(dataloader[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()
                output = self.client_models[str(i)](data)
                output_array = output.detach().cpu().numpy()
                output_class = np.argmax(output_array, axis=-1)
                target_array = target.detach().cpu().numpy()
                intermediate_accuracy += np.sum(output_class == target_array)
            accuracy = intermediate_accuracy / length[str(i)]* 100
            total += length[str(i)]

            self.accuracy_list.append(accuracy)
            accuracies[i] = intermediate_accuracy
        overall_accuracy = np.sum(accuracies) / total * 100
        return overall_accuracy

    def loop(self, epochs, experiment_name, k):

        loss_tests = []
        loss_trains = []
        loss_tests2 = []
        loss_trains2 = []
        accuracies = []
        accuracies_train = []
        best_accuracy = 0

        self.selected_clients_arr = np.zeros((epochs, self.total_clients, self.total_clients))
        self.init_ifca(k)

        for i in range(epochs):
            print(i)
            list1 = []
            self.selected_clients = [x for x in range(self.total_clients)]
            self.iteration = i
            if i != 0: 
                self.cluster_ifca(self.selected_clients,k)
            self.run_ifca(self.selected_clients)
            self.combine_ifca(self.selected_clients,k)

            loss_train, loss_test = self.update_local_models(self.selected_clients)
            
            loss_tests.append(loss_test.detach().cpu().numpy())
            loss_trains.append(loss_train.detach().cpu().numpy())
            
            accuracy_val = self.calc_accuracy(self.dataloaders_test, self.len_test)
            print('val accuracy: ', accuracy_val)

            accuracy = self.calc_accuracy(self.dataloaders_really_test, self.len_really_test)
            print('test accuracy: ', accuracy)
            accuracies.append(accuracy)
            if accuracy_val > best_accuracy:
                print(best_accuracy)
                print('accuracy is best accuracy')
                print(self.accuracy_list)
                best_accuracy = accuracy_val

                # save all of this in a .txt file
                fname = os.path.join('checkpoints_bandits', experiment_name, 'test_accuracies.txt')
                np.savetxt(fname, self.accuracy_list)
                fname = os.path.join('checkpoints_bandits', experiment_name, 'test_accuracy.txt')
                np.savetxt(fname, [accuracy])


        # print(self.phis)
        fname = os.path.join('checkpoints_bandits', experiment_name, 'accuracies.txt')
        np.savetxt(fname, accuracies)

        plt.figure()
        plt.plot(loss_trains, label='train loss before')
        plt.plot(loss_tests, label='test loss before')
        plt.plot(loss_trains2, label='train loss after')
        plt.plot(loss_tests2, label='test loss after')
        plt.title('loss curve')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join('checkpoints_bandits', experiment_name, 'loss_curve.png'))
        plt.clf()
        plt.plot(accuracies, label='test')
        plt.plot(accuracies_train, label='train')
        plt.title('accuracy progression')
        plt.legend()
        plt.savefig(os.path.join('checkpoints_bandits', experiment_name, 'accuracy_progression.png'))




if __name__ == '__main__':
    settings, save_dir = init()
    print(settings)

    train_data, test_data = tk.datasets.mnist.load_data()
    instance = Partition_MNIST_NIID(train_data[0], train_data[1])
    classes_per_user = settings['n_classes_per_user']
    total_clients = 100 
    n_classes_total = settings['n_classes_total']
    train_partition = instance.create_partition(n_classes_total, classes_per_user, total_clients)
    test_instance = Partition_MNIST_NIID(test_data[0], test_data[1])
    test_partition = test_instance.create_partition_test(instance.sample_array)

     # split train dataset into train and val
    print([len(x) for x in train_partition.values()])
    fraction = 0.8
    #print([int(np.floor(len(x)*0.8)) for x in train_partition.values()])
    train_length = [int(np.floor(len(x)*0.8)) for x in train_partition.values()]

    train_partition2 = {}
    val_partition = {}

    for key in train_partition.keys():
        og_length = len(train_partition[key])
        og_samples = np.array(train_partition[key])
        list1 = [x for x in range(og_length)]
        train_samples = np.random.choice(list1,size=int(np.floor(og_length*fraction)),replace=False)
        val_samples = [x for x in range(og_length) if x not in train_samples]
        train_partition2[key] = og_samples[train_samples]
        val_partition[key] = og_samples[val_samples]
    print([len(x) for x in train_partition2.values()])
    print([len(x) for x in val_partition.values()])

    k = settings['k']
    seed = settings['seed']
    epochs = settings['n_epochs']
    experiment_name = settings['experiment_name']
    data_fraction = settings['data_fraction']
    print(experiment_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    p2p = IFCA(total_clients, train_data, train_partition2, val_partition, 
            test_data, test_partition,settings['n_clients_UCB'], 
            settings['alpha'],test='IFCA')
    p2p.loop(epochs, experiment_name, k)