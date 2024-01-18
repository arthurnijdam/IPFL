import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from scipy.special import binom
from sklearn.metrics import balanced_accuracy_score

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
import os
import sys
from os.path import join as osj
from bisect import bisect
from collections import defaultdict
import pickle
import json
import wfdb


def read_dict_beats():
    with open(DICT_BEATS, "rb") as f:
        return pickle.load(f)


def read_data_beats():
    with open(DATA_BEATS, "rb") as f:
        return pickle.load(f)


def ensure_normalized_and_detrended(beats):
    for key in beats.keys():
        b = beats[key]["beats"]
        if not np.allclose(np.linalg.norm(b, axis=1, ord=2), 1):
            raise AssertionError(f"Beats of patient {key} is not normalized.")

        p = np.polyfit(np.arange(b.shape[1]), b.T, deg=1)
        if not np.allclose(p, 0):
            raise AssertionError(f"Beats of patient {key} is not detrended.")


def get_paced_patients(patient_ids):
    paced = []
    for id_ in patient_ids:
        annotation = wfdb.rdann(osj(DATA_ROOT, str(id_)), extension='atr')
        labels = np.unique(annotation.symbol)
        if ("/" in labels):
            paced.append(id_)
    return np.array(paced)




# patient_ids = pd.read_csv(osj("..", "files", "patient_ids.csv"), header=None).to_numpy().reshape(-1)
# paced_patients = pd.read_csv(osj("..", "files", "paced_patients.csv"), header=None).to_numpy().reshape(-1)
# excluded_patients = pd.read_csv(osj("..", "files", "excluded_patients.csv"), header=None).to_numpy().reshape(-1)
def get_base_model(in_channels):
    """
    Returns the model from paper: Personalized Monitoring and Advance Warning System for Cardiac Arrhythmias.
    """
    # Input size: 128x1
    # 128x1 -> 122x32 -> 40x32 -> 34x16 -> 11x16 -> 5x16 -> 1x16
    model = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=7, padding=0, bias=True),
        nn.MaxPool1d(3),
        nn.Tanh(),

        nn.Conv1d(32, 16, kernel_size=7, padding=0, bias=True),
        nn.MaxPool1d(3),
        nn.Tanh(),

        nn.Conv1d(16, 16, kernel_size=7, padding=0, bias=True),
        nn.MaxPool1d(3),
        nn.Tanh(),

        nn.Flatten(),

        nn.Linear(16, 32, bias=True),
        nn.ReLU(),

        nn.Linear(32, 2, bias=True),

    )
    return model


# AAMI standards: only use the first 5 seconds, only use 'healthy' heartbeats for training.
# They combine each client's dataset with other clients' datasets (with domain Adaptation)
# and test on the other 25 seconds + the abnormal heartbeats of the client.


def train_test_split(data_beats, seconds=5, data_fraction=1):
    data_beats_train = {}
    data_beats_val = {}
    data_beats_test = {}
    for i in data_beats.keys():
        data_beats_train[i] = {'class': None, 'beats': None}
        data_beats_val[i] = {'class': None, 'beats': None}
        data_beats_test[i] = {'class': None, 'beats': None}

    for patient in data_beats.keys():
        length_train = int(np.ceil(len(data_beats[patient]['beats']) * (seconds / 30)))  # only take first 5 seconds

        random_test = np.arange(int(np.ceil(len(data_beats[patient]['beats']))))
        random_val= np.arange(length_train)
        for ii in random_val:
            random_test = np.delete(random_test,ii)

        # Data fraction, take part of the data
        random_val = np.random.choice(random_val, size=int(np.ceil(data_fraction* length_train)), replace=False)

        random_train = np.random.choice(random_val, size=int(np.ceil(0.8 *data_fraction* length_train)), replace=False)
        for ii in random_train:
            index = np.where(random_val == ii)[0]
            random_val = np.delete(random_val, index)


        #random_val = np.arange(int(np.ceil(0.8 * length)))
        #random_train = np.random.choice(random_val, size=int(np.ceil(0.8 * 0.8 * length)), replace=False)
        #for ii in random_train:
        #    index = np.where(random_val == ii)[0]
        #    random_val = np.delete(random_val, index)

        data_beats_train[patient]['class'] = data_beats[patient]['class'][np.sort(random_train)]
        data_beats_test[patient]['class'] = data_beats[patient]['class'][random_test]
        data_beats_val[patient]['class'] = data_beats[patient]['class'][random_val]
        data_beats_train[patient]['beats'] = data_beats[patient]['beats'][np.sort(random_train)]
        data_beats_test[patient]['beats'] = data_beats[patient]['beats'][random_test]
        data_beats_val[patient]['beats'] = data_beats[patient]['beats'][random_val]

    return data_beats_train, data_beats_val, data_beats_test


import copy

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





class MIT_BIH(Dataset):
    def __init__(self, patients, data):
        self.patients = patients
        self.data = data
        self.to_one_dataset()

    def to_one_dataset(self):
        data_vector = torch.zeros(self.__len__(), 128)
        labels_vector = torch.zeros(self.__len__())
        k = 0
        for i, patient in enumerate(self.patients):
            data_vector[k:k + len(self.data[patient]['beats']), :] = torch.from_numpy(self.data[patient]['beats'])
            classes = copy.deepcopy(self.data[patient]['class'])
            indices = classes == 'N'
            indices2 = classes != 'N'
            classes[indices] = 0
            classes[indices2] = 1
            classes = np.array(classes, dtype='int')
            labels_vector[k:k + len(self.data[patient]['beats'])] = torch.from_numpy(classes)
            k += len(self.data[patient]['beats'])
        self.y = labels_vector.long()
        self.X = data_vector.double()

    def __len__(self):
        length_total = 0
        for patient in self.patients:
            length_total += len(self.data[patient]['beats'])
        # print(len(self.data[patient]['beats']))
        return length_total

    def __getitem__(self, idx):
        return (self.X[idx, :], self.y[idx])


class P2P_AFPL():

    def __init__(self, patients_left, train_data, val_data,test_data, n_clients_selected,test='local'):
        self.selected_clients = patients_left
        self.network = get_base_model(1)
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
        if self.test == 'AFPL':
            self.client_models_global = {}

        if self.test == 'bandits':
            self.comb_UCB = combinatorial_UCB(self.total_clients,n_clients_selected)

        for idx, i in enumerate(self.patients_left):
            self.client_models[str(idx)] = copy.deepcopy(self.network).double().cuda()
            self.optimizers[str(idx)] = torch.optim.SGD(self.client_models[str(idx)].parameters(), lr=0.01,
                                                        momentum=0.5)
            dataset_train = MIT_BIH([self.patients_left[idx]], train_data)
            self.len[str(idx)] = len(dataset_train)
            self.dataloaders[str(idx)] = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

            dataset_test = MIT_BIH([self.patients_left[idx]], val_data)
            self.len_test[str(idx)] = len(dataset_test)
            self.dataloaders_test[str(idx)] = DataLoader(dataset_test, batch_size=32, shuffle=False)
            self.best_test_loss[str(idx)] = 10000000
            self.current_test_loss[str(idx)] = 100000
            self.current_train_loss[str(idx)] = 1000000
            if self.test == 'AFPL':
                self.client_models_global[str(idx)] = copy.deepcopy(self.network).double().cuda()
                self.shared_model = copy.deepcopy(self.network).double().cuda()

            dataset_really_test = MIT_BIH([self.patients_left[idx]], test_data)
            self.len_really_test[str(idx)] = len(dataset_really_test)
            self.dataloaders_really_test[str(idx)] = DataLoader(dataset_really_test, batch_size=32, shuffle=False)
        self.dataset_train = dataset_train

    def update_local_models(self, selected_clients):
        self.dw = {}
        loss_test = 0
        loss_test2 = 0
        losses = 0
        losses2 = 0
        loss_test3 = 0
        losses3 = 0

        for idx, i in enumerate(selected_clients):

            dataloader = self.dataloaders[str(i)]
            optimizer = torch.optim.Adam(self.client_models[str(i)].parameters(), lr=0.001 * 0.95 ** self.iteration)
            self.client_models[str(i)].train()

            if self.test == 'AFPL':
                self.client_models_global[str(i)] = copy.deepcopy(self.shared_model)
                self.client_models_global[str(i)].train()
                optimizer_global = torch.optim.Adam(self.client_models_global[str(i)].parameters(),
                                                    lr=0.001 * 0.95 ** self.iteration)

            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
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
                    data = data.double().unsqueeze(1).cuda()
                    target = target.long().cuda()
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
                    data = data.double().unsqueeze(1).cuda()
                    target = target.long().cuda()
                    output = self.client_models[str(i)](data)
                    output = F.log_softmax(output, dim=-1)

                    loss_test2 += F.nll_loss(output, target)

            losses2 += loss_test2 / self.len[str(i)]
            self.current_train_loss[str(i)] = loss_test2 / self.len[str(i)]

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
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
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
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
                output = self.shared_model(data)
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
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
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
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
                output = self.client_models[str(i)](data)
                output = F.log_softmax(output, dim=-1)

                loss_test2 += F.nll_loss(output, target).detach().cpu().numpy()

            loss_test2 = loss_test2 / self.len[str(i)]
            losses2 += loss_test2

        self.shared_model = self.shared_model.double()
        return losses, losses2

    def calc_accuracy(self, dataloaders, length):
        accuracies = np.zeros(len(self.selected_clients))
        total = 0
        self.accuracy_list = []
        preds = []
        trues = []
        for i in self.selected_clients:
            # dataloader = self.dataloaders_really_test[str(i)]
            dataloader = dataloaders[str(i)]
            intermediate_accuracy = 0
            self.client_models[str(i)].eval()
            y_pred = []
            y_true = []
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
                output = self.client_models[str(i)](data)
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

            #self.accuracy_list.append(accuracy)
            for sub in y_pred:
                for j in sub:
                    preds.append(j)
            for sub in y_true:
                for j in sub:
                    trues.append(j)
            #preds.append([j for sub in y_pred for j in sub])
            pred = np.array([j for sub in y_pred for j in sub])
            true = np.array([j for sub in y_true for j in sub])
            #print(i)

            #print(pred)
            #print(true)
            #print(balanced_accuracy_score(true, pred))
            self.accuracy_list.append(balanced_accuracy_score(true,pred))

            C = confusion_matrix(true, pred).ravel()
            if len(C) == 4:
                df = pandas.DataFrame([[C[3], C[1]], [C[2], C[0]]], columns=['Positive', 'Negative'],
                                      index=['Predicted Positive', 'Predicted Negative'])

            total += length[str(i)]
            accuracies[i] = intermediate_accuracy

        overall_accuracy = np.sum(accuracies) / total * 100
        #print()

       # breakpoint()

        return balanced_accuracy_score(trues,preds)#overall_accuracy

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
                data = data.unsqueeze(1).double().cuda()
                target = target.long().cuda()
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
                data = data.unsqueeze(1).double().cuda()
                target = target.long().cuda()
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
                data = data.unsqueeze(1).double().cuda()
                target = target.long().cuda()
                output = self.client_models[str(client)](data)
                output = F.log_softmax(output, dim=-1)
                loss_test2 += F.nll_loss(output, target).detach().cpu().numpy()

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
                data = data.unsqueeze(1).double().cuda()
                target = target.long().cuda()
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
                data = data.unsqueeze(1).double().cuda()
                target = target.long().cuda()
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
                data = data.unsqueeze(1).double().cuda()
                target = target.long().cuda()
                output2 = self.client_models[str(client)](data)
                output2 = F.log_softmax(output2, dim=-1)

                loss_test2 += F.nll_loss(output2, target).detach().cpu().numpy()

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

            loss_train, loss_test = self.update_local_models(self.selected_clients)
            loss_tests.append(loss_test.detach().cpu().numpy())
            loss_trains.append(loss_train.detach().cpu().numpy())

            if self.test == 'AFPL':
                losses2, losses3 = self.AFPL()

            if self.test == 'local':
                print('we are done')

            if self.test == 'federated':
                losses2, losses3 = self.federated_averaging()

            if self.test == 'bandits':
                losses2 = 0
                losses3 = 0
                for client in range(self.total_clients):
                    loss_test2, loss_train2, selected_clients2,selected_clients_UCB = self.bandits(client, i)
                    losses2 += loss_test2
                    if len(selected_clients2) < 1:
                        losses3 += self.current_train_loss[str(client)].detach().cpu().numpy()
                    else:
                        losses3 += loss_train2
                    self.phis[client, selected_clients2] += 1
                    self.phisUCB[client,selected_clients_UCB] += 1
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

            if self.test != 'local':
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

import yaml
import os
import shutil
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



settings, save_dir = init()
print(save_dir)
if __name__ == '__main__':
    DATA_ROOT = osj("/mimer/NOBACKUP/groups/snic2022-22-122/arthur/", "dataset_beats")
    DICT_BEATS = osj(DATA_ROOT, "5min_normal_beats.pkl")
    DATA_BEATS = osj(DATA_ROOT, "30min_beats.pkl")

    DATA_ROOT = "/mimer/NOBACKUP/groups/snic2022-22-122/arthur/physionet.org/files/mitdb/1.0.0/"
    RECORDS = osj(DATA_ROOT, "RECORDS")
    #print(RECORDS)
    patient_ids = pd.read_csv(RECORDS, header=None).to_numpy().reshape(-1)
    #print(patient_ids)
    paced_patients = get_paced_patients(patient_ids)
    excluded_patients = np.array([105, 114, 201, 202, 207, 209, 213, 222, 223, 234])  # according to paper
    #print(np.concatenate((paced_patients, excluded_patients)))

    dict_beats = read_dict_beats()
    data_beats = read_data_beats()
    ensure_normalized_and_detrended(dict_beats)
    ensure_normalized_and_detrended(data_beats)

    # print(np.shape(dict_beats))
    import collections

    # print(dict_beats.keys())
    patients_out = np.concatenate((paced_patients, excluded_patients))
    #print(patients_out)
    patients_left = list(copy.deepcopy(patient_ids))

    for idx, i in enumerate(patient_ids):
        if i in patients_out:
            patients_left.remove(i)

    #print(patients_left)

    # print(dict_beats[101]['beats'])
    # print(dict_beats[101]['class'])
    labels = ['N', 'V', 'S', 'Q', 'F']
    dictionary = {}
    for i in labels:
        dictionary[i] = 0

    list1 = []
    array = np.zeros((len(patients_left), 2))
    for idx, i in enumerate(patients_left):
        print(len(data_beats[i]['class']))
        list1.append(data_beats[i]['class'])
        counter = collections.Counter(data_beats[i]['class'])
        for j in counter.keys():
            dictionary[j] += counter[j]
            if j == 'N':
                array[idx, 0] += counter[j]
            else:
                array[idx, 1] += counter[j]

    seconds = 5
    data_beats_train, data_beats_val, data_beats_test = train_test_split(data_beats, seconds,settings['data_fraction'])

    mit_bih = MIT_BIH(patients_left, data_beats_train)
    x_sample, y_sample = mit_bih.__getitem__(0)
    dataloader = DataLoader(mit_bih, batch_size=32, shuffle=True, num_workers=0)
    mit_bih_test = MIT_BIH(patients_left, data_beats_val)
    x_sample, y_sample = mit_bih_test.__getitem__(0)
    dataloader_test = DataLoader(mit_bih_test, batch_size=32, shuffle=False, num_workers=0)

    import collections
    from time import time
    import random
    from sklearn.metrics import f1_score, confusion_matrix
    import pandas

    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    random.seed(settings['seed'])
    torch.cuda.manual_seed_all(settings['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    experiment_name = settings['experiment_name']
    test = settings['type']
    n_epochs = settings['n_epochs']
    p2p = P2P_AFPL(patients_left, data_beats_train, data_beats_val,data_beats_test,settings['n_clients_UCB'], test)
    alphas = p2p.loop(n_epochs, p2p, experiment_name)




