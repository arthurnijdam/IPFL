import numpy as np
import pandas as pd
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

# Combinatorial UCB
import math


class combinatorial_UCB(object):
    def __init__(self, n_clients, n_clients_selected=10, algorithm='UCB1_tuned'):
        self.n_clients = n_clients

        # define variables for storage
        # which clients we select
        self.times_selected = np.zeros((n_clients, n_clients))  # to record how often each client got selected
        self.reward_per_client = np.zeros((n_clients, n_clients))  # to record what reward we collected per client
        self.reward2_per_client = np.zeros(
            (n_clients, n_clients))  # to record the squared reward per client (needed for UCB1-tuned)
        # how many clients we select
        self.n_clients_selected_arr = []
        self.reward3_per_client = np.zeros((n_clients, n_clients - 1))
        self.times_selected2 = np.zeros((n_clients, n_clients - 1))

        if n_clients_selected == None:
            self.n_clients_selected = np.zeros((n_clients, 1))
        else:
            self.n_clients_selected = np.ones((n_clients, 1)) * n_clients_selected

        self.algorithm = algorithm

    def UCB(self, this_client, n):
        # for this_client in range(self.n_clients):
        other_clients = [x for x in range(self.n_clients) if x != this_client[0]]

        upper_bound = np.zeros(self.n_clients)
        for i, other_client in enumerate(other_clients):
            if self.times_selected[this_client, other_client] == 0:  # make first iteration value high
                upper_bound[other_client] = 1e500
            else:
                # We first calculate the average reward gained for this client
                average_reward = self.reward_per_client[this_client, other_client] / self.times_selected[
                    this_client, other_client]

                # Then we compute the confidence interval [avg_reward - delta, avg_reward + delta]
                if self.algorithm == 'UCB1':
                    delta = math.sqrt(2 * math.log(n) / self.times_selected[this_client, other_client])

                if self.algorithm == 'UCB1_tuned':
                    variance_bound = self.reward2_per_client[this_client, other_client] / self.times_selected[
                        this_client, other_client] - average_reward ** 2
                    variance_bound += math.sqrt(2 * math.log(n) / self.times_selected[this_client, other_client])

                    factor = np.min([variance_bound, 1 / 4])
                    delta = math.sqrt(factor * math.log(n) / self.times_selected[this_client, other_client])

                # upper bound
                upper_bound[other_client] = average_reward + delta

        if self.algorithm == 'random':
            upper_bound = np.random.rand(self.n_clients)

        # select the client with the highest upper bound
        sorted_upper_bound = np.flip(np.argsort(upper_bound))

        # if epoch == 0:
        #     n_clients_selected = self.n_clients -2

        # else:
        n_clients_selected = self.n_clients_selected[i]-1

        # Run UCB again to determine the number of clients
        # upper_bound2 = np.zeros(self.n_clients-1)
        # for ii in range(1,self.n_clients-1):
        #    if self.times_selected2[this_client,ii]==0: # make first iteration value high
        #        upper_bound2[ii] = 1e500
        #        n_clients_selected = self.n_clients -2
        #    else:
        # predict the reward when selecting these clients
        #        average_reward_n_clients = self.reward3_per_client[this_client,ii] / self.times_selected2[this_client,ii]
        #        delta = math.sqrt(2*math.log(n)) / np.sum(self.times_selected2[this_client,ii])

        #        upper_bound2[ii] = average_reward_n_clients + delta

        #        n_clients_selected = np.argmax(upper_bound2)

        # n_clients_selected_arr.append(n_clients_selected)
        selected_clients = sorted_upper_bound[:int(n_clients_selected + 1)]

        self.times_selected[this_client, selected_clients] += 1
        return selected_clients

    def collect_reward(self, this_client, selected_clients, observations):
        # collect the reward
        reward = observations[selected_clients]  # df.iloc[n,selected_client]
        self.reward_per_client[this_client, selected_clients] += reward
        self.reward2_per_client[this_client, selected_clients] += reward ** 2

        # reward for numbers of clients selected
        # n_clients_selected = len(selected_clients)-1
        # self.times_selected2[this_client,n_clients_selected] += 1

    # if epoch == 0:
    #     self.n_clients_selected[this_client] = np.sum(observations)
    # reward2 = np.abs(n_clients_selected - np.sum(observations))
    # self.reward3_per_client[this_client,n_clients_selected] += 1 - reward2 / self.n_clients

    def to_client(self, this_client, n):
        self.selected_clients = self.UCB(this_client, n)
        return self.selected_clients

    def to_server(self, this_client, observation):
        self.collect_reward(this_client, self.selected_clients, observation)



class P2P_AFPL():
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
        if self.test == 'AFPL':
            self.client_models_global = {}
            self.alpha = alpha

        if self.test == 'bandits':
            self.comb_UCB = combinatorial_UCB(self.total_clients,n_clients_selected)

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
            if self.test == 'AFPL':
                self.client_models_global[str(i)] = copy.deepcopy(self.network).double().cuda()
                self.shared_model = copy.deepcopy(self.network).double().cuda()
        self.dataset_train = dataset_train
        self.dataloader_centralized = DataLoader(dataset, batch_size=32, shuffle=True)

    def update_local_models(self ,selected_clients):
        self.dw = {}
        loss_test = 0
        loss_test2 = 0
        losses = 0
        losses2 = 0
        loss_test3 = 0
        losses3 = 0

        for idx ,i in enumerate(selected_clients):

            dataloader = self.dataloaders[str(i)]
            optimizer= torch.optim.Adam(self.client_models[str(i)].parameters() ,lr=0.001 *0.95**self.iteration)
            self.client_models[str(i)].train()

            if self.test == 'AFPL':
                self.client_models_global[str(i)] = copy.deepcopy(self.shared_model)
                self.client_models_global[str(i)].train()
                optimizer_global = torch.optim.Adam(self.client_models_global[str(i)].parameters()
                                                    ,lr=0.001 *0.95**self.iteration)

            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target =target.long().cuda()

                optimizer.zero_grad()
                output = self.client_models[str(i)](data)
                loss = F.nll_loss(output ,target)

                if self.test == 'AFPL':
                    optimizer_global.zero_grad()
                    output_global= self.client_models_global[str(i)](data)
                    loss_global = F.nll_loss(output_global ,target)
                    loss_global.backward()
                    optimizer_global.step()

                loss.backward()
                optimizer.step()

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

    def centralized(self ,selected_clients):
        self.dw = {}
        loss_test = 0
        loss_test2 = 0
        losses = 0
        losses2 = 0
        loss_test3 = 0
        losses3 = 0

        dataloader = self.dataloader_centralized
        optimizer= torch.optim.Adam(self.client_models[str(0)].parameters() ,lr=0.001 *0.95**self.iteration)
        self.client_models[str(0)].train()


        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.double().cuda()
            target =target.long().cuda()

            optimizer.zero_grad()
            output = self.client_models[str(0)](data)
            loss = F.nll_loss(output ,target)

            loss.backward()
            optimizer.step()

        self.client_models[str(0)].eval()
        for idx, i in enumerate(selected_clients):
            dataloader_test = self.dataloaders_test[str(i)]
            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.double().cuda()
                    target =target.long().cuda()

                    output = self.client_models[str(0)](data)
                    loss_test += F.nll_loss(output ,target)

                losses += loss_test / self.len_test[str(i)]
        self.current_test_loss[str(0)] = loss_test /self.len_test[str(0)]
        if self.current_test_loss[str(0)] < self.best_test_loss[str(0)]:
            torch.save(self.client_models[str(0)].state_dict(), os.path.join(save_dir, 'model', 'best_model ' +str(0 ) +'.pt'))
            self.best_test_loss[str(0)] = self.current_test_loss[str(0)]


        loss_test2 = 0
        self.client_models[str(0)].eval()
        for idx, i in enumerate(selected_clients):
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.dataloaders[str(i)]):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = self.client_models[str(0)](data)
                    loss_test2 += F.nll_loss(output, target)

                losses2 += loss_test2 / self.len[str(i)]
        self.current_train_loss[str(i)] = loss_test2 / self.len[str(i)]

        print('full train loss: ', losses2)
        print('full loss: ', losses)
        for idx, i in enumerate(selected_clients):
            if i != 0:
                for (name, param), (name2, param2) in zip(self.client_models[str(i)].named_parameters(),
                                                          self.client_models[str(0)].named_parameters()):
                    param.data = param2.data
                self.client_models[str(i)].double()

        return losses2, losses

    def combine_models(self, i, client_numbers, set_as=True):
        zero_copy = copy.deepcopy(self.client_models[str(i)])  # This is used to collect the model in
        j = 0
        client_numbers_plus_client = np.concatenate((client_numbers, np.array([int(i)])))  # This is more efficient
        #  alphas = zero_copy.alphas.detach()
        # alphas[i] = 1 - torch.sum(
        #     torch.tensor([iii for idx, iii in enumerate(alphas) if idx != i and idx in client_numbers]))
        # It's not possible to set the value of self.alphas[i], so instead we determine it manually here
        alphas = torch.ones(len(client_numbers_plus_client)).cuda() / (len(client_numbers_plus_client))
        # print(alphas)
        for ii in client_numbers_plus_client:
            #  print(ii)
            for (name, param), (name2, param2) in zip(zero_copy.named_parameters(), self.client_models[
                str(ii)].named_parameters()):  # self.client_models[str(ii)].named_parameters()):

                if name != 'alphas':
                    if j == 0:
                        param.data = torch.zeros(param.shape).cuda()

                    param.data += alphas[j] * param2.data  # we add all participating client's models to the one here.

            j += 1

        # self.client_models[str(i)] = zero_copy.double()
        if set_as == True:
            for (name, param), (name2, param2) in zip(self.client_models[str(i)].named_parameters(),
                                                      zero_copy.named_parameters()):
                param.data = param2.data
            self.client_models[str(i)].double()
        else:
            return zero_copy.double()

    def federated_averaging(self):
        self.shared_model = copy.deepcopy(self.network).double().cuda()
        n_clients = len(self.selected_clients)
        weight = [self.len[str(x)] for x in self.selected_clients]
        weight = weight / np.sum(weight)

        losses = 0
        losses2 = 0
        # print("weights ",weight)
        for idx, i in enumerate(self.selected_clients):
            for (name, param), (name2, param2) in zip(self.shared_model.named_parameters()
                    , self.client_models[str(i)].named_parameters()):
                if idx == 0:
                    param.data = torch.zeros(param.shape).cuda().double()
                param.data += weight[idx] * param2.data

        self.shared_model = self.shared_model.double().eval()

        for i in self.selected_clients:
            self.client_models[str(i)] = copy.deepcopy(self.shared_model)  # copy global model to the clients
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(self.shared_model(data), target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(i)]
            losses += loss_test
            if loss_test < self.best_test_loss[str(i)]:
                torch.save(self.client_models[str(i)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(i)] = loss_test
            self.client_models[str(i)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test2 += F.nll_loss(self.shared_model(data), target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(i)]
            losses2 += loss_test2

        return losses, losses2

    def federated_averaging2(self):
        # Accumulate global model
        self.shared_model = copy.deepcopy(self.network).double().cuda()
        n_clients = len(self.selected_clients)
        weight = [self.len[str(x)] for x in self.selected_clients]
        weight = weight / np.sum(weight)

        losses = 0
        losses2 = 0
        # print("weights ",weight)
        for idx, i in enumerate(self.selected_clients):
            for (name, param), (name2, param2) in zip(self.shared_model.named_parameters()
                    , self.client_models[str(i)].named_parameters()):
                if idx == 0:
                    param.data = torch.zeros(param.shape).cuda().double()
                param.data += weight[idx] * param2.data

        self.shared_model = self.shared_model.double().eval()

        # loop over clients
        for i in self.selected_clients:
            # check if model improves performance
            #self.client_models[str(i)] = copy.deepcopy(self.shared_model)  # copy global model to the clients
            self.client_models[str(i)].eval().cuda()
            loss_test = 0
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(self.shared_model(data), target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(self.client_models[str(i)](data), target).detach().cpu().numpy()

            ey = loss_test / self.len_test[str(i)]
            current_test = loss_test2 / self.len_test[str(i)]
            if ey < current_test:
               # print('replaced local model with global')
                self.client_models[str(i)] = copy.deepcopy(self.shared_model)
                losses += ey
                if ey < self.best_test_loss[str(i)]:
                    torch.save(self.client_models[str(i)].state_dict(),
                               os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                    self.best_test_loss[str(i)] = ey
            else:
              #  print('nothing')
                losses += current_test
                if current_test < self.best_test_loss[str(i)]:
                    torch.save(self.client_models[str(i)].state_dict(),
                               os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                    self.best_test_loss[str(i)] = current_test


            self.client_models[str(i)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test2 += F.nll_loss(self.client_models[str(i)](data), target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(i)]
        losses2 += loss_test2
        return losses, losses2

    def AFPL(self):  # use alpha = 0.25 = 0.75 global model + 0.25 local model
        self.shared_model_old = copy.deepcopy(self.shared_model)
        self.shared_model = copy.deepcopy(self.network).double().cuda()
        n_clients = len(self.selected_clients)
        weight = [self.len[str(x)] for x in self.selected_clients]
        weight = weight / np.sum(weight)

        losses = 0
        losses2 = 0

        # accumulate local weights
        for idx, i in enumerate(self.selected_clients):
            for (name, param), (name2, param2), (name3, param3), (name4, param4) in zip(
                    self.shared_model.named_parameters()
                    , self.client_models_global[str(i)].named_parameters(),
                    self.shared_model_old.named_parameters(),
                    self.client_models[str(i)].named_parameters()):
                if idx == 0:
                    param.data = torch.zeros(param.shape).cuda().double()
                param.data += weight[idx] * param2.data  # accumulate local weights
                param4.data = self.alpha * param4.data + (1-self.alpha) * param3.data  # do AFPL local model update: note that we take the previous global model
            self.client_models[str(i)] = self.client_models[str(i)].double()
            self.client_models[str(i)].eval()
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(self.client_models[str(i)](data), target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(i)]
            losses += loss_test
            if loss_test < self.best_test_loss[str(i)]:
                torch.save(self.client_models[str(i)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(i)] = loss_test
            self.client_models[str(i)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(i)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test2 += F.nll_loss(self.client_models[str(i)](data), target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(i)]
            losses2 += loss_test2

        self.shared_model = self.shared_model.double()
        return losses, losses2

    def optimal_fedavg(self):
        losses = 0
        losses2 = 0
        for i in range(self.total_clients):

            self.client_models[str(i)].eval()
            dataloader_test = self.dataloaders_test[str(i)]
            loss_test = 0
            loss_test2 = 0
            # print(np.where(adj_matrix[i,:]>0)[0])
            label_informed_selected_clients = np.where(adj_matrix[i, :] > 0)[0]

            label_informed_shared_model = self.combine_models(i, label_informed_selected_clients, set_as=False)
            label_informed_shared_model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output2 = label_informed_shared_model(data)

                    loss_test += F.nll_loss(output2, target)
                if loss_test / self.len_test[str(i)] < self.best_test_loss[str(i)]:
                    torch.save(self.client_models[str(i)].state_dict(),
                               os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                    self.best_test_loss[str(i)] = loss_test / self.len_test[str(i)]
            losses += loss_test / self.len_test[str(i)]

            dataloader_test = self.dataloaders[str(i)]
            loss_test2 = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = label_informed_shared_model(data)
                    loss_test2 += F.nll_loss(output, target)
            losses2 += loss_test2 / self.len[str(i)]

            self.combine_models(i, label_informed_selected_clients)
        return losses, losses2

    def my_method(self, client):

        selected_clients = []
        other_clients = [x for x in range(self.total_clients) if x is not client]
        ey = np.zeros(len(other_clients))  # fix indices
        current_test = np.zeros(len(other_clients))
        collected_clients = []
        list1 = np.arange(len(other_clients))
        np.random.shuffle(list1)
        for i in list1:
            # selected_clients_coalition = other_clients[i] +[client]
            shared_model = self.combine_models(client, [other_clients[i]], set_as=False)

            if len(collected_clients) > 0:
                all_clients = collected_clients + [other_clients[i]]
                shared_model2 = self.combine_models(client, all_clients, set_as=False)

            # print(selected_clients_coalition)
            shared_model.eval().cuda()
            self.client_models[str(client)].eval().cuda()
            loss_test = 0
            loss_test2 = 0
            loss_test3 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(shared_model(data), target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

                if len(collected_clients) > 0:
                    loss_test3 += F.nll_loss(shared_model2(data), target).detach().cpu().numpy()

            ey[i] = loss_test / self.len_test[str(client)]
            current_test[i] = loss_test2 / self.len_test[str(client)]
            if ey[i] < current_test[i]:
                if len(collected_clients) > 0:
                    test2 = loss_test3 / self.len_test[str(client)]
                    if test2 < current_test[i]:
                        collected_clients.append(other_clients[i])
                else:
                    collected_clients.append(other_clients[i])
        loss_test = current_test[i]

        selected_clients = np.where(ey <= self.current_test_loss[str(client)].detach().cpu().numpy())[0]
        selected_clients = [other_clients[x] for x in selected_clients]
        selected_clients = collected_clients

        if len(selected_clients) > 0:
            # self.client_models[str(client)] = copy.deepcopy(shared_model)
            self.combine_models(client, selected_clients, set_as=True)
            # self.client_models[str(client)].double().eval().cuda()
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(client)]
            if loss_test < self.best_test_loss[str(client)]:
                torch.save(self.client_models[str(client)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(client)] = loss_test
            self.client_models[str(client)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test2 += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(client)]

        #    print('test loss: ',loss_test)
        # return ey, selected_clients
        return loss_test, loss_test2, selected_clients

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

    def my_method2(self, client, k=10):

        selected_clients = []
        other_clients = [x for x in range(self.total_clients) if x is not client]
        ey = np.zeros(len(other_clients))  # fix indices
        current_test = np.zeros(len(other_clients))
        collected_clients = []
        list1 = np.arange(len(other_clients))
        np.random.shuffle(list1)
        for i in list1[:k]:
            shared_model = self.combine_models(client, [other_clients[i]], set_as=False)

            if len(collected_clients) > 0:
                all_clients = collected_clients + [other_clients[i]]
                shared_model2 = self.combine_models(client, all_clients, set_as=False)

            shared_model.eval().cuda()
            self.client_models[str(client)].eval().cuda()
            loss_test = 0
            loss_test2 = 0
            loss_test3 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(shared_model(data), target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

                if len(collected_clients) > 0:
                    loss_test3 += F.nll_loss(shared_model2(data), target).detach().cpu().numpy()

            ey[i] = loss_test / self.len_test[str(client)]
            current_test[i] = loss_test2 / self.len_test[str(client)]
            if ey[i] < current_test[i]:
                if len(collected_clients) > 0:
                    test2 = loss_test3 / self.len_test[str(client)]
                    if test2 < current_test[i]:
                        collected_clients.append(other_clients[i])
                else:
                    collected_clients.append(other_clients[i])
        loss_test = current_test[i]

        selected_clients = np.where(ey <= self.current_test_loss[str(client)].detach().cpu().numpy())[0]
        selected_clients = [other_clients[x] for x in selected_clients]
        selected_clients = collected_clients

        if len(selected_clients) > 0:
            self.combine_models(client, selected_clients, set_as=True)
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(client)]
            if loss_test < self.best_test_loss[str(client)]:
                torch.save(self.client_models[str(client)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(client)] = loss_test
            self.client_models[str(client)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test2 += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(client)]
        return loss_test, loss_test2, selected_clients

    def bandits(self, client, n):

        selected_clients = []
        other_clients = [x for x in range(self.total_clients) if x != client]
        # print(other_clients)
        ey = np.zeros(self.total_clients)  # fix indices
        current_test = np.zeros(self.total_clients)
        collected_clients = []

        selected_clients_UCB = self.comb_UCB.to_client([client], n)
        if client == 1:
            print('selected clients UCB: ', selected_clients_UCB)
        for i in selected_clients_UCB:
            shared_model = self.combine_models(client, [i], set_as=False)

            if len(collected_clients) > 0:
                all_clients = collected_clients + [i]
                shared_model2 = self.combine_models(client, all_clients, set_as=False)

            shared_model.eval().cuda()
            self.client_models[str(client)].eval().cuda()
            loss_test = 0
            loss_test2 = 0
            loss_test3 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(shared_model(data), target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

                if len(collected_clients) > 0:
                    loss_test3 += F.nll_loss(shared_model2(data), target).detach().cpu().numpy()

            ey[i] = loss_test / self.len_test[str(client)]
            current_test[i] = loss_test2 / self.len_test[str(client)]
            if ey[i] < current_test[i]:
                if len(collected_clients) > 0:
                    test2 = loss_test3 / self.len_test[str(client)]
                    if test2 < current_test[i]:
                        collected_clients.append(i)
                else:
                    collected_clients.append(i)
        loss_test = current_test[i]
        selected_clients = np.where(ey <= self.current_test_loss[str(client)].detach().cpu().numpy())[0]
        # selected_clients = [other_clients[x] for x in selected_clients]

        selected_clients = collected_clients

        observation = np.zeros(self.total_clients)
        observation[selected_clients] = 1
        if client == 1:
            print(observation)

        self.comb_UCB.to_server(client, observation)

        if len(selected_clients) > 0:
            self.combine_models(client, selected_clients, set_as=True)
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(client)]
            if loss_test < self.best_test_loss[str(client)]:
                torch.save(self.client_models[str(client)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(client)] = loss_test
            self.client_models[str(client)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(client)]):
                data = data.double().cuda()
                target = target.long().cuda()

                loss_test2 += F.nll_loss(self.client_models[str(client)](data), target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(client)]
        return loss_test, loss_test2, selected_clients, selected_clients_UCB

    def loop(self, epochs, p2p, experiment_name):

        loss_tests = []
        loss_trains = []
        loss_tests2 = []
        loss_trains2 = []
        accuracies = []
        accuracies_train = []
        best_accuracy = 0
        self.p2p = p2p
        self.phis = np.zeros((self.total_clients, self.total_clients))
        self.phisUCB = np.zeros((self.total_clients, self.total_clients))
        self.selected_clients_arr = np.zeros((epochs, self.total_clients, self.total_clients))

        for i in range(epochs):
            print(i)
            self.iteration = i
            list1 = []
            self.selected_clients = [x for x in range(self.total_clients)]

            if self.test != 'centralized':
                loss_train, loss_test = self.update_local_models(self.selected_clients)
            else:
                loss_train, loss_test = self.centralized(self.selected_clients)
            loss_tests.append(loss_test.detach().cpu().numpy())
            loss_trains.append(loss_train.detach().cpu().numpy())

            if self.test == 'AFPL':
                losses2, losses3 = self.AFPL()

            if self.test == 'prtfl':
                losses2, losses3 = self.federated_averaging2()

            if self.test == 'local':
                print('we are done')

            if self.test == 'federated':
                losses2, losses3 = self.federated_averaging()

            if self.test == 'bandits':
                losses2 = 0
                losses3 = 0
                for client in range(self.total_clients):
                    loss_test2, loss_train2, selected_clients2, selected_clients_UCB= self.bandits(client, i)
                    losses2 += loss_test2
                    if len(selected_clients2) < 1:
                        losses3 += self.current_train_loss[str(client)].detach().cpu().numpy()
                    else:
                        losses3 += loss_train2
                    self.phis[client, selected_clients2] += 1
                    self.selected_clients_arr[i, client, selected_clients2] += 1
                    self.phisUCB[client, selected_clients_UCB] += 1
                fname = os.path.join('checkpoints_bandits', experiment_name, 'phi' + str(i) + '.txt')
                np.savetxt(fname, self.phis)
                fname = os.path.join('checkpoints_bandits', experiment_name, 'phi_UCB' + str(i) + '.txt')
                np.savetxt(fname, self.phisUCB)

            if self.test == 'mine':
                losses2 = 0
                losses3 = 0
                for client in range(self.total_clients):
                    loss_test2, loss_train2, selected_clients2 = self.my_method2(client)
                    losses2 += loss_test2
                    if len(selected_clients2) < 1:
                        losses3 += self.current_train_loss[str(client)].detach().cpu().numpy()

                    else:
                        losses3 += loss_train2
                    self.phis[client, selected_clients2] += 1
                fname = os.path.join('checkpoints_bandits', experiment_name, 'phi' + str(i) + '.txt')
                np.savetxt(fname, self.phis)

            if self.test == 'optimal':
                losses2, losses3 = self.optimal_fedavg()
                losses2 = losses2.detach().cpu().numpy()
                losses3 = losses3.detach().cpu().numpy()

            if self.test != 'local' and self.test != 'centralized':
                print('loss after my code: ', losses2)
                print('train loss after my code: ', losses3)
                loss_tests2.append(losses2)
                loss_trains2.append(losses3)
                fname = os.path.join('checkpoints_bandits', experiment_name, 'losses_test.txt')
                np.savetxt(fname, loss_tests2)
                fname = os.path.join('checkpoints_bandits', experiment_name, 'losses_train.txt')
                np.savetxt(fname, loss_trains2)


            else:
                fname = os.path.join('checkpoints_bandits', experiment_name, 'losses_test.txt')
                np.savetxt(fname, loss_tests)
                fname = os.path.join('checkpoints_bandits', experiment_name, 'losses_train.txt')
                np.savetxt(fname, loss_trains)

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
            # accuracy_train = self.calc_accuracy(test=False)
            # print(accuracy_train)
            # accuracies_train.append(accuracy_train)
            # print(self.phis)

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

if __name__ == '__main__':

    settings, save_dir = init()
    print(settings)

    from models import Net
    from data import FEMNIST_dataset, Partition_MNIST_NIID
    from data import MNIST_NIID_dataset
    import tensorflow.keras as tk
    #import yaml
    import random

    ### Divide MNIST dataset over a total of 10 clients. Each client gets 2 classes.
    train_data, test_data = tk.datasets.mnist.load_data()
    data_fraction = int(settings['data_fraction']*60000)
    print('data_fraction: ',data_fraction)
    blub = np.random.randint(len(train_data[0]),size=data_fraction)
    instance = Partition_MNIST_NIID(train_data[0][blub],train_data[1][blub])
    #instance = Partition_MNIST_NIID(train_data[0],train_data[1])
    classes_per_user = settings['n_classes_per_user']
    total_clients = settings['n_clients']
    #total_clients = 100
    n_classes_total = settings['n_classes_total']
    train_partition = instance.create_partition(n_classes_total,classes_per_user,total_clients)
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

    # who do we expect to collaborate based on labels?
    import networkx as nx


    def Repeat(x):
        _size = len(x)
        repeated = []
        cit = {}
        for i in range(_size):
            k = i + 1
            key = str(np.sort(x[i, :]))
            if key not in cit:
                cit[key] = [i]

            for j in range(k, _size):
                if x[i, 0] == x[j, 0] and x[i, 1] == x[j, 1]:  # and x[i,:] not in repeated:
                    repeated.append(x[i, :])
                    if j not in cit[key]:
                        cit[key].append(j)
                if x[i, 0] == x[j, 1] and x[i, 1] == x[j, 0]:  # and x[i,:] and x[j,:] not in repeated:
                    repeated.append(x[i, :])
                    if j not in cit[key]:
                        cit[key].append(j)

        return cit


    cit = Repeat(instance.sample_array)
    GS = nx.DiGraph()
    adj_matrix = np.zeros((total_clients, total_clients))
    for p in cit.keys():
        arr = cit[p]
        for i in arr:
           for j in arr:
                if i != j:
                    GS.add_edge(i, j)
                    adj_matrix[i, j] += 1
    nx.draw(GS, with_labels=True)
    print(adj_matrix)

    import collections
    from time import time
    import random

    seed = settings['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test = settings['type']
    print(test)
    p2p = P2P_AFPL(total_clients, train_data, train_partition2, val_partition, test_data, test_partition,settings['n_clients_UCB'], settings['alpha'],test)
    phis = p2p.loop(settings['n_epochs'], p2p, settings['experiment_name'])













