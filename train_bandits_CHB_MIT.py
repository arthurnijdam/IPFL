### We first load the dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import h5py
import torch.nn.functional as F
from torchinfo import summary

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

def calc_table(label_array,output_array):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i,ele in enumerate(output_array):
        for j,ele2 in enumerate(ele):
            if ele2 == 1 and label_array[i][j] == 1:
                TP += 1
            if ele2 == 0 and label_array[i][j] ==0:
                TN += 1
            if ele2 == 1 and label_array[i][j] ==0:
                FP += 1
            if ele2 ==0 and label_array[i][j] == 1:
                FN += 1

    print('TP: ',TP)
    print('TN: ',TN)
    print('FP: ',FP)
    print('FN: ',FN)


# Split the seizures and allow to define the train ratio.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, i, iteration, partition='train'):
        self.iteration = iteration
        self.filepath = filepath
        self.partition = partition
        self.number = i
        images, labels = self.load_data()

        images, self.labels = self.train_val_test(images, labels)
        self.dataset = self.normalize(images)
        self.size = len(self.labels)

    def load_data(self):
        arrays = {}
        filepath = self.filepath + 'chb' + self.create_digits(self.number) + '_4s_0s.mat'
        f = h5py.File(filepath)
        index = 0
        for k, v in f.items():
            arrays[index] = np.array(v)
            index = index + 1
        all_electrodes = np.transpose(arrays[0])
        labels = np.transpose(arrays[1])
        image = np.reshape(all_electrodes, (np.shape(all_electrodes)[0], -1, 1024))
        selected_electrodes = [1, 13]
        image = image[:, selected_electrodes, :]
        return image, labels

    def train_val_test(self, images, labels):

        # find the begin, end and the number of seizures
        begin = 0
        n_seizures = 0
        seizure_onset = []
        seizure_end = []
        seizures = []
        non_seizures = []
        for ii in range(len(labels)):
            if labels[ii] == 1 and begin == 0:  # check where the seizure begins
                n_seizures += 1
                seizure_onset.append(ii)
            if labels[ii] == 0 and begin == 1:  # check where the seizure ends
                seizure_end.append(ii - 1)
            if labels[ii] == 1:
                seizures.append(ii)
                if ii == len(labels) - 1:
                    seizure_end.append(ii)
            if labels[ii] == 0:
                non_seizures.append(ii)
            begin = labels[ii]

        # if it is even, put half of them in the train,
        # and half of them in the test dataset

        train_idx = []
        test_idx = []
        # if it is odd, split one of the seizures and divide the rest.
        seizures_train = np.floor(n_seizures / 2)

        for j in range(n_seizures):
            if j < int(seizures_train):
                train_idx.append([x for x in range(seizure_onset[j], seizure_end[j] + 1)])
            else:
                if n_seizures % 2 == 0:
                    test_idx.append([x for x in range(seizure_onset[j], seizure_end[j] + 1)])
                else:
                    if j == int(seizures_train):  # split seizure into 2
                        train_idx.append([x for x in range(seizure_onset[j],
                                                           int(np.ceil(seizure_onset[j] + 0.5 * (
                                                                       seizure_end[j] - seizure_onset[j]))))])
                        test_idx.append([x for x in range(
                            int(np.ceil(seizure_onset[j] + 0.5 * (seizure_end[j] - seizure_onset[j]))),
                            seizure_end[j] + 1)])
                    else:
                        test_idx.append([x for x in range(seizure_onset[j], seizure_end[j] + 1)])

        train_idx = np.sort(np.hstack(
            [np.hstack(train_idx), [x for idx, x in enumerate(non_seizures) if idx < int(len(non_seizures) / 2)]]))
        test_idx = np.sort(np.hstack(
            [np.hstack(test_idx), [x for idx, x in enumerate(non_seizures) if idx >= int(len(non_seizures) / 2)]]))

        # plt.plot(train_idx,labels[train_idx],'r.')
        # plt.plot(test_idx,labels[test_idx],'b.')
        np.random.seed(seed)
        train_idx= np.random.choice(train_idx, size=int(np.ceil(data_fraction * len(train_idx))), replace=False)

        train_data = images[train_idx, :, :]
        test_data = images[test_idx, :, :]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # train_data, test_data, train_labels, test_labels = train_test_split(images,labels,random_state=1)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                          test_size=0.5,
                                                                          random_state=self.iteration)

        if self.partition == 'train':
            return train_data, train_labels

        # return images[:fraction_train,:,:], labels[:fraction_train]
        if self.partition == 'val':
            return val_data, val_labels

        #  return images[fraction_train:fraction_val,:,:],labels[fraction_train:fraction_val]
        if self.partition == 'test':
            return test_data, test_labels

        #  return images[fraction_val:,:,:],labels[fraction_val:]

    def create_digits(self, number):
        if number < 10:
            return '0' + str(number)
        else:
            return str(number)

    def normalize(self, data):
        input_shape = np.shape(data)
        data = np.reshape(data, (-1, 1024))
        var = np.mean(data, axis=0)
        mean = np.mean(data, axis=0)
        normalized = (data - mean) / var
        normalized = np.reshape(data, (input_shape[0], 2, 1024))
        return normalized

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index, :, :]), torch.from_numpy(self.labels[index])

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.labels)

# network
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=3))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=3))
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2, stride=3))

        self.fc = nn.Linear(64 * 15, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


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
    def __init__(self, patients_left, n_clients_selected, test='local'):
        self.selected_clients = patients_left
        self.network = ConvNet(2).to(device).double()
        self.best_test_loss = {}
        self.best_test_loss_global = 1000000
        self.current_test_loss = {}
        self.current_train_loss = {}
        self.test = test
        self.total_clients = len(self.selected_clients)
        self.patients_left = patients_left
        self.client_models = {}
        self.optimizers = {}
        self.dataloaders = {}
        self.len = {}
        self.len_test = {}
        self.len_really_test = {}
        self.dataloaders_test = {}
        self.dataloaders_really_test = {}
        filepath = '/mimer/NOBACKUP/groups/naiss2024-22-903/IPFL_data/'

        if self.test == 'AFPL':
            self.client_models_global = {}

        if self.test == 'bandits':
            self.comb_UCB = combinatorial_UCB(self.total_clients, n_clients_selected)

        for idx, i in enumerate(self.patients_left):
            self.client_models[str(idx)] = copy.deepcopy(self.network).double().cuda()
            self.optimizers[str(idx)] = torch.optim.SGD(self.client_models[str(idx)].parameters(), lr=0.01,
                                                        momentum=0.5)

            if idx == 1:
                dataset = torch.utils.data.ConcatDataset([dataset_train, CustomDataset(filepath, self.patients_left[idx], 0, 'train')])
            if idx > 1 :
                dataset = torch.utils.data.ConcatDataset(
                    [dataset,  CustomDataset(filepath, self.patients_left[idx], 0, 'train')])


            dataset_train = CustomDataset(filepath, self.patients_left[idx], 0, 'train')
            self.len[str(idx)] = len(dataset_train)
            self.dataloaders[str(idx)] = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

            dataset_test = CustomDataset(filepath, self.patients_left[idx], 0, 'val')
            self.len_test[str(idx)] = len(dataset_test)
            self.dataloaders_test[str(idx)] = DataLoader(dataset_test, batch_size=32, shuffle=False)
            self.best_test_loss[str(idx)] = 10000000
            self.current_test_loss[str(idx)] = 100000
            self.current_train_loss[str(idx)] = 1000000
            if self.test == 'AFPL':
                self.client_models_global[str(idx)] = copy.deepcopy(self.network).double().cuda()
                self.shared_model = copy.deepcopy(self.network).double().cuda()

            dataset_really_test = CustomDataset(filepath, self.patients_left[idx], 0, 'test')
            self.len_really_test[str(idx)] = len(dataset_really_test)
            self.dataloaders_really_test[str(idx)] = DataLoader(dataset_really_test, batch_size=32, shuffle=False)
        self.dataset_train = dataset_train
        self.dataloader_centralized = DataLoader(dataset, batch_size=32, shuffle=True)

    def update_local_models(self, selected_clients):
        self.dw = {}
        loss_test = 0
        loss_test2 = 0
        losses = 0
        losses2 = 0
        loss_test3 = 0
        losses3 = 0
        losses4 = 0

        for idx, i in enumerate(selected_clients):
            if self.iteration == 1000:
                dataset_train = CustomDataset(filepath, i + 1, self.iteration, 'train')
                self.dataloaders[str(i)] = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

                dataset_test = CustomDataset(filepath, i + 1, self.iteration, 'val')
                self.dataloaders_test[str(i)] = DataLoader(dataset_test, batch_size=32, shuffle=False)

            dataloader = self.dataloaders[str(i)]
            optimizer = torch.optim.Adam(self.client_models[str(i)].parameters(), lr=0.005 * 0.95 ** self.iteration)
            self.client_models[str(i)].train()

            if self.test == 'AFPL':
                self.client_models_global[str(i)] = copy.deepcopy(self.shared_model)
                self.client_models_global[str(i)].train()
                optimizer_global = torch.optim.Adam(self.client_models_global[str(i)].parameters(),
                                                    lr=0.005 * 0.95 ** self.iteration)

            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(i)](data)
                output = F.log_softmax(output, dim=-1)
                # data = data.double().cuda()
                # target=target.long().cuda()

                optimizer.zero_grad()
                # output = self.client_models[str(i)](data)
                loss = F.nll_loss(output, target)

                if self.test == 'AFPL':
                    optimizer_global.zero_grad()
                    output_global = self.client_models_global[str(i)](data)
                    loss_global = F.nll_loss(output_global, target)
                    loss_global.backward()
                    optimizer_global.step()

                loss.backward()
                optimizer.step()

            self.client_models[str(i)].eval()
            dataloader_test = self.dataloaders_test[str(i)]
            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))
                    output = self.client_models[str(i)](data)
                    output = F.log_softmax(output, dim=-1)

                    loss_test += F.nll_loss(output, target)
                self.current_test_loss[str(i)] = loss_test / self.len_test[str(i)]
                if self.current_test_loss[str(i)] < self.best_test_loss[str(i)]:
                    torch.save(self.client_models[str(i)].state_dict(),
                               os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                    self.best_test_loss[str(i)] = self.current_test_loss[str(i)]

            losses += loss_test / self.len_test[str(i)]
            loss_test2 = 0
            self.client_models[str(i)].eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))
                    output = self.client_models[str(i)](data)
                    output = F.log_softmax(output, dim=-1)

                    loss_test2 += F.nll_loss(output, target)

            losses2 += loss_test2 / self.len[str(i)]
            self.current_train_loss[str(i)] = loss_test2 / self.len[str(i)]
        print('losses before: ', losses4)
        print('full train loss: ', losses2)
        print('full loss: ', losses)

        return losses2, losses

    def centralized(self, selected_clients):
        self.dw = {}
        loss_test = 0
        loss_test2 = 0
        losses = 0
        losses2 = 0
        loss_test3 = 0
        losses3 = 0
        losses4 = 0


        dataloader = self.dataloader_centralized
        b = 1
        optimizer = torch.optim.Adam(self.client_models[str(b)].parameters(), lr=0.005 * 0.95 ** self.iteration)
        self.client_models[str(b)].train()

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.cuda()
            target = target.type(torch.LongTensor).squeeze(1).cuda()
            # Forward pass
            data = torch.reshape(data, (-1, 1, 2 * 1024))
            output = self.client_models[str(b)](data)
            output = F.log_softmax(output, dim=-1)
            # data = data.double().cuda()
            # target=target.long().cuda()

            optimizer.zero_grad()
            # output = self.client_models[str(i)](data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

        self.client_models[str(b)].eval()
        for idx, i in enumerate(selected_clients):
            dataloader_test = self.dataloaders_test[str(i)]
            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))
                    output = self.client_models[str(b)](data)
                    output = F.log_softmax(output, dim=-1)

                    loss_test += F.nll_loss(output, target)
            losses += loss_test / self.len_test[str(i)]
        self.current_test_loss[str(b)] = loss_test / self.len_test[str(b)]
        if self.current_test_loss[str(b)] < self.best_test_loss[str(b)]:
                torch.save(self.client_models[str(b)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(b) + '.pt'))
                self.best_test_loss[str(b)] = self.current_test_loss[str(b)]


        loss_test2 = 0
        for idx, i in enumerate(selected_clients):
            self.client_models[str(b)].eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.dataloaders[str(b)]):
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))
                    output = self.client_models[str(b)](data)
                    output = F.log_softmax(output, dim=-1)

                    loss_test2 += F.nll_loss(output, target)

            losses2 += loss_test2 / self.len[str(i)]
        for idx, i in enumerate(selected_clients):
            if i != b:
                for (name, param), (name2, param2) in zip(self.client_models[str(i)].named_parameters(),
                                                          self.client_models[str(b)].named_parameters()):
                    param.data = param2.data
                self.client_models[str(i)].double()

        print('full train loss: ', losses2)
        print('full loss: ', losses)

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
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.shared_model(data)
                output = F.log_softmax(output, dim=-1)

                loss_test += F.nll_loss(output, target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(i)]
            losses += loss_test
            if loss_test < self.best_test_loss[str(i)]:
                torch.save(self.client_models[str(i)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(i)] = loss_test
            self.client_models[str(i)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(i)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.shared_model(data)
                output = F.log_softmax(output, dim=-1)

                loss_test2 += F.nll_loss(output, target).detach().cpu().numpy()

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
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.shared_model(data)
                output = F.log_softmax(output, dim=-1)
                output2 = self.client_models[str(i)](data)
                output2 = F.log_softmax(output2, dim=-1)
                loss_test += F.nll_loss(output, target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(output2, target).detach().cpu().numpy()

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
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(i)](data)
                output = F.log_softmax(output, dim=-1)

                loss_test2 += F.nll_loss(output, target).detach().cpu().numpy()

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
                param4.data = 0.25 * param4.data + 0.75 * param3.data  # do AFPL local model update: note that we take the previous global model
            self.client_models[str(i)] = self.client_models[str(i)].double()
            self.client_models[str(i)].eval()
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(i)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(i)](data)
                output = F.log_softmax(output, dim=-1)

                loss_test += F.nll_loss(output, target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(i)]
            losses += loss_test
            if loss_test < self.best_test_loss[str(i)]:
                torch.save(self.client_models[str(i)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(i)] = loss_test
            self.client_models[str(i)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(i)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(i)](data)
                output = F.log_softmax(output, dim=-1)

                loss_test2 += F.nll_loss(output, target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(i)]
            losses2 += loss_test2

        self.shared_model = self.shared_model.double()
        return losses, losses2

    def my_method2(self, client, k=30):

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
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = shared_model(data)
                output = F.log_softmax(output, dim=-1)
                local_output = self.client_models[str(client)](data)
                local_output = F.log_softmax(local_output, dim=-1)

                loss_test += F.nll_loss(output, target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(local_output, target).detach().cpu().numpy()

                if len(collected_clients) > 0:
                    output2 = shared_model2(data)
                    output2 = F.log_softmax(output2, dim=-1)
                    loss_test3 += F.nll_loss(output2, target).detach().cpu().numpy()

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
        # print(client)
        # print(loss_test)
        # print(self.current_test_loss[str(client)])
        # print(ey)

        selected_clients = np.where(ey <= self.current_test_loss[str(client)].detach().cpu().numpy())[0]
        selected_clients = [other_clients[x] for x in selected_clients]
        # print(selected_clients)
        selected_clients = collected_clients

        if len(selected_clients) > 0:
            self.combine_models(client, selected_clients, set_as=True)
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(client)](data)
                output = F.log_softmax(output, dim=-1)

                loss_test += F.nll_loss(output, target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(client)]
            if loss_test < self.best_test_loss[str(client)]:
                torch.save(self.client_models[str(client)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(client)] = loss_test
            self.client_models[str(client)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(client)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(client)](data)
                output = F.log_softmax(output, dim=-1)
                loss_test2 += F.nll_loss(output, target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(client)]
        return loss_test, loss_test2, selected_clients

    def bandits(self, client, n, length):

        selected_clients = []
        other_clients = [x for x in range(self.total_clients) if x != client]
        # print(other_clients)
        ey = np.zeros(self.total_clients)  # fix indices
        current_test = np.zeros(self.total_clients)
        collected_clients = []
        # print('START OF UCB CLIENT: ',client)
        selected_clients_UCB = self.comb_UCB.to_client([client], n)
        if client == 0:
            print('selected clients UCB: ', selected_clients_UCB)
        old_accuracy = 0
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

            accuracy_shared = 0
            accuracy_local = 0

            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = shared_model(data)
                output = F.log_softmax(output, dim=-1)
                output2 = self.client_models[str(client)](data)
                output2 = F.log_softmax(output2, dim=-1)
                loss_test += F.nll_loss(output, target).detach().cpu().numpy()
                loss_test2 += F.nll_loss(output2, target).detach().cpu().numpy()

                if len(collected_clients) > 0:
                    output = shared_model2(data)
                    output = F.log_softmax(output, dim=-1)
                    loss_test3 += F.nll_loss(output, target).detach().cpu().numpy()
                    old_accuracy = accuracy_shareds

                # calculate accuracy
                output_array = output.detach().cpu().numpy()
                output_class = np.argmax(output_array, axis=-1)
                target_array = target.detach().cpu().numpy()
                accuracy_shared += np.sum(output_class == target_array)

                output_array2 = output2.detach().cpu().numpy()
                output_class2 = np.argmax(output_array2, axis=-1)
                accuracy_local += np.sum(output_class2 == target_array)

            accuracy_locals = accuracy_local / length[str(client)] * 100
            accuracy_shareds = accuracy_shared / length[str(client)] * 100
            # print('CLIENT: ',i)
            #  print(accuracy_locals)
            # print(accuracy_shareds)

            # ACCURACY-based client selection
            if accuracy_shareds > accuracy_locals and accuracy_shareds > old_accuracy:
                collected_clients.append(i)

                # LOSS-BASED client selection:
        #   ey[i] = loss_test / self.len_test[str(client)]
        #   current_test[i] = loss_test2 / self.len_test[str(client)]
        #   if ey[i] < current_test[i]:
        #        if len(collected_clients) > 0:
        #            test2 = loss_test3 / self.len_test[str(client)]
        #            if test2 < current_test[i]:
        #               collected_clients.append(i)
        #        else:
        #           collected_clients.append(i)

        loss_test = current_test[i]
        # selected_clients = np.where(ey <= self.current_test_loss[str(client)].detach().cpu().numpy())[0]

        selected_clients = collected_clients

        observation = np.zeros(self.total_clients)
        observation[selected_clients] = 1
        if client == 0:
            print(observation)

        self.comb_UCB.to_server(client, observation)

        if len(selected_clients) > 0:
            self.combine_models(client, selected_clients, set_as=True)
            loss_test = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders_test[str(client)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output2 = self.client_models[str(client)](data)
                output2 = F.log_softmax(output2, dim=-1)

                loss_test += F.nll_loss(output2, target).detach().cpu().numpy()

            loss_test = loss_test / self.len_test[str(client)]
            if loss_test < self.best_test_loss[str(client)]:
                torch.save(self.client_models[str(client)].state_dict(),
                           os.path.join(save_dir, 'model', 'best_model' + str(i) + '.pt'))
                self.best_test_loss[str(client)] = loss_test
            self.client_models[str(client)].eval()
            loss_test2 = 0
            for batch_idx, (data, target) in enumerate(self.dataloaders[str(client)]):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output2 = self.client_models[str(client)](data)
                output2 = F.log_softmax(output2, dim=-1)

                loss_test2 += F.nll_loss(output2, target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(client)]
        return loss_test, loss_test2, selected_clients, selected_clients_UCB

    def calc_accuracy(self, dataloaders, length):
        accuracies = np.zeros(len(self.selected_clients))
        total = 0
        self.accuracy_list = []
        for i in self.selected_clients:
            if self.test == 'centralized':
                b = 1
            else:
                b = i
                # dataloader = self.dataloaders_really_test[str(i)]
            dataloader = dataloaders[str(i)]
            intermediate_accuracy = 0
            self.client_models[str(b)].eval()
            y_pred = []
            y_true = []
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
                output = self.client_models[str(b)](data)
                output = F.log_softmax(output, dim=-1)
                output_array = output.detach().cpu().numpy()
                output_class = np.argmax(output_array, axis=-1)
                target_array = target.detach().cpu().numpy()
                intermediate_accuracy += np.sum(output_class == target_array)
                y_pred.append(list(output_class))
                y_true.append(list(target_array))

            # accuracy = intermediate_accuracy / p2p.len_really_test[str(i)] * 100
            # print(i)
            accuracy = intermediate_accuracy / length[str(i)] * 100
            #if self.iteration % 5 == 0:
            #    print('client accuracy : ', str(i))
             #   print(accuracy)
            self.accuracy_list.append(accuracy)

            pred = np.array([j for sub in y_pred for j in sub])
            true = np.array([j for sub in y_true for j in sub])
            C = confusion_matrix(true, pred).ravel()
            if len(C) == 4:
                df = pandas.DataFrame([[C[3], C[1]], [C[2], C[0]]], columns=['Positive', 'Negative'],
                                      index=['Predicted Positive', 'Predicted Negative'])

            total += length[str(i)]
            accuracies[i] = intermediate_accuracy
        overall_accuracy = np.sum(accuracies) / total * 100

        return overall_accuracy



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
            # print(self.client_models[str(1)].layer1[0].weight.detach().cpu().numpy())
            self.iteration = i
            list1 = []
            self.selected_clients = [x for x in range(self.total_clients)]
            if self.test != 'centralized':
                loss_train, loss_test = self.update_local_models(self.selected_clients)
            else:
                loss_train, loss_test = self.centralized(self.selected_clients)

            loss_tests.append(loss_test.detach().cpu().numpy())
            loss_trains.append(loss_train.detach().cpu().numpy())
            # print(self.client_models[str(1)].layer1[0].weight.detach().cpu().numpy())
            accuracy_val = self.calc_accuracy(self.dataloaders_test, self.len_test)
            print('val accuracy before bandits: ', accuracy_val)
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
                    loss_test2, loss_train2, selected_clients2, selected_clients_UCB = self.bandits(client, i,
                                                                                                    self.len_test)
                    losses2 += loss_test2
                    if len(selected_clients2) < 1:
                        losses3 += self.current_train_loss[str(client)].detach().cpu().numpy()
                    else:
                        losses3 += loss_train2
                    self.phis[client, selected_clients2] += 1
                    self.phisUCB[client, selected_clients_UCB] += 1
                    self.selected_clients_arr[i, client, selected_clients2] += 1
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
                    # print(selected_clients2)
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
        # plt.plot(accuracies_train,label='train')
        plt.title('accuracy progression')
        plt.legend()
        plt.savefig(os.path.join('checkpoints_bandits', experiment_name, 'accuracy_progression.png'))
        return accuracies

if __name__ == '__main__':
    settings, save_dir = init()
    import random
    from sklearn.metrics import confusion_matrix
    import pandas

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = settings['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    experiment_name = settings['experiment_name']
    test = settings['type'] #local'
    data_fraction = settings['data_fraction']
    n_epochs = settings['n_epochs']
    patients_removed = [6, 14, 16]
    patients_left = [x for x in range(1, 24) ] #if x not in patients_removed]
    print(patients_left)
    p2p = P2P_AFPL(patients_left, settings['n_clients_UCB'], test)
    accuracies_fed = p2p.loop(n_epochs, p2p, experiment_name)
