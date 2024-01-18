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
import yaml 

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

        nn.Linear(32, 3, bias=True),

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
import sklearn
import warnings
warnings.filterwarnings("ignore")

class MIT_BIH(Dataset):
    def __init__(self, patients, data):
        self.patients = patients
        self.data = data
        self.to_one_dataset()

    def to_one_dataset(self):
        length_total = 0
        for patient in self.patients:
            length_total += len(self.data[patient]['beats'])
        # print(len(self.data[patient]['beats']))
        data_vector = torch.zeros(length_total, 128)
        labels_vector = torch.zeros(length_total)
        k = 0
        for i, patient in enumerate(self.patients):
            data_vector[k:k + len(self.data[patient]['beats']), :] = torch.from_numpy(self.data[patient]['beats'])
            classes = copy.deepcopy(self.data[patient]['class'])
            indices = classes == 'N'
            indices2 = classes == 'S'
            indices3 = classes == 'V'
            indices4 = classes == 'F'
            indices5 = classes == 'Q'

            classes[indices] = 0
            classes[indices2] = 1
            classes[indices3] = 2
            classes[indices4] = 2  # classify F as V
            classes[indices5] = 3
            classes = np.array(classes, dtype='int')
            labels_vector[k:k + len(self.data[patient]['beats'])] = torch.from_numpy(classes)
            k += len(self.data[patient]['beats'])
        # remove q entries
        indices6 = np.array(labels_vector != 3)
        self.y = torch.masked_select(labels_vector, torch.from_numpy(indices6)).long()
        self.X = data_vector[indices6, :].double()

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx, :], self.y[idx])


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

import collections
from time import time
import random
from sklearn.metrics import f1_score, confusion_matrix
import pandas

class IFCA():
    def __init__(self, patients_left, train_data, test_data, real_test_data, test='local'):
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
        self.dataloaders_test = {}
        self.dataloaders_really_test = {}
        self.len_really_test = {}
        
        for idx, i in enumerate(self.patients_left):
            self.client_models[str(idx)] = copy.deepcopy(self.network).double().cuda()
            self.optimizers[str(idx)] = torch.optim.SGD(self.client_models[str(idx)].parameters(), lr=0.01,   momentum=0.5)

            if idx == 1:
                dataset = torch.utils.data.ConcatDataset([dataset_train,MIT_BIH([self.patients_left[idx]], train_data)])
            if idx > 1 :
                dataset = torch.utils.data.ConcatDataset(
                    [dataset, MIT_BIH([self.patients_left[idx]], train_data)])

            dataset_train = MIT_BIH([self.patients_left[idx]], train_data)
            self.len[str(idx)] = len(dataset_train)
            self.dataloaders[str(idx)] = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)

            dataset_test = MIT_BIH([self.patients_left[idx]], test_data)
            self.len_test[str(idx)] = len(dataset_test)
            self.dataloaders_test[str(idx)] = DataLoader(dataset_test, batch_size=32, shuffle=False)

            dataset_really_test = MIT_BIH([self.patients_left[idx]], real_test_data)
            self.len_really_test[str(idx)] = len(dataset_really_test)
            self.dataloaders_really_test[str(idx)] = DataLoader(dataset_really_test, batch_size=32, shuffle=True,
                                                                num_workers=0)

            self.best_test_loss[str(idx)] = 10000000
            self.current_test_loss[str(idx)] = 100000
            self.current_train_loss[str(idx)] = 1000000
            if self.test == 'AFPL':
                self.client_models_global[str(idx)] = copy.deepcopy(self.network).double().cuda()
                self.shared_model = copy.deepcopy(self.network).double().cuda()
        self.dataset_train = dataset_train
        self.dataloader_centralized = DataLoader(dataset, batch_size=32,shuffle=True)
       
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
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
                output = model(data)
                output = F.log_softmax(output, dim=-1)

                optimizer.zero_grad()
                # output = self.client_models[str(i)](data)
                loss = F.nll_loss(output, target)

                loss.backward()
                optimizer.step()
            
            # save the trained model at the client side 
            self.client_models[str(i)] = copy.deepcopy(model)
            
    def cluster_ifca(self, selected_clients, k): 
        
        losses = np.zeros((len(selected_clients),k))
        for idx, i in enumerate(selected_clients): 
            dataloader = self.dataloaders[str(i)]
            for k_i in range(k): 
                model = copy.deepcopy(self.global_models_ifca[str(k_i)])
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.double().unsqueeze(1).cuda()
                    target = target.long().cuda()
                    output = model(data)
                    output = F.log_softmax(output, dim=-1)

                    loss = F.nll_loss(output ,target)
                    losses[i,k_i] += loss 
                    
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
                    data = data.double().unsqueeze(1).cuda()
                    target = target.long().cuda()

                    output = self.client_models[str(i)](data)
                    output = F.log_softmax(output, dim=-1)

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

    def calc_accuracy(self, dataloaders, length):
        accuracies = np.zeros(len(self.selected_clients))
        total = 0
        self.accuracy_list = []
        preds = []
        trues = []
        for i in self.selected_clients:
            # dataloader = self.dataloaders_really_test[str(i)]
            if self.test == 'centralized':
                b = 0
            else:
                b = i
            dataloader = dataloaders[str(i)]
            intermediate_accuracy = 0
            self.client_models[str(b)].eval()
            y_pred = []
            y_true = []
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().unsqueeze(1).cuda()
                target = target.long().cuda()
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

    
    def calc_accuracy2(self, dataloader, length):
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

    DATA_ROOT = "/mimer/NOBACKUP/groups/naiss2023-22-980/arthur/code/Federated_Averaging/data"
    DATA_BEATS = osj(DATA_ROOT, "30min_beats.pkl")
    data_fraction = settings['data_fraction']
    RECORDS = osj(DATA_ROOT, "RECORDS")
    print(RECORDS)
    patient_ids = pd.read_csv(RECORDS, header=None).to_numpy().reshape(-1)
    print(patient_ids)
    data_beats = read_data_beats()
    ensure_normalized_and_detrended(data_beats)
    paced_patients = np.array([102, 104, 107, 217])
    excluded_patients = np.array([])  # np.array([105, 114, 201, 202,207, 209, 213, 222, 223, 234]) # according to paper
    print(np.concatenate((paced_patients, excluded_patients)))
    n_bandits_UCB = settings['n_clients_UCB']

    patients_out = np.concatenate((paced_patients, excluded_patients))
    print(patients_out)
    patients_left = list(copy.deepcopy(patient_ids))

    for idx, i in enumerate(patient_ids):
        if i in patients_out:
            patients_left.remove(i)

    seconds = 5
    data_beats_train, data_beats_val, data_beats_test = train_test_split(data_beats, seconds=5,data_fraction=data_fraction)

    # Load supraventricular dataset:
    DATA_ROOT = "/mimer/NOBACKUP/groups/naiss2023-22-980/arthur/code/Federated_Averaging/data"
    DATA_BEATS = osj(DATA_ROOT, "30min_beats_supraventricular.pkl")

    RECORDS = osj(DATA_ROOT, "RECORDS_S")
    patient_ids_sup = pd.read_csv(RECORDS, header=None).to_numpy().reshape(-1)
    data_beats_sup = read_data_beats()
    ensure_normalized_and_detrended(data_beats_sup)
    seconds = 5
    data_beats_train_sup, data_beats_val_sup, data_beats_test_sup = train_test_split(data_beats_sup, data_fraction=data_fraction,seconds=5)
    # concatenate the datasets
    data_beats_tr = {}
    data_beats_tr.update(data_beats_train)
    data_beats_tr.update(data_beats_train_sup)

    data_beats_v = {}
    data_beats_v.update(data_beats_val)
    data_beats_v.update(data_beats_val_sup)

    data_beats_t = {}
    data_beats_t.update(data_beats_test)
    data_beats_t.update(data_beats_test_sup)
    patients_left = [x for x in list(data_beats_tr.keys()) if x not in paced_patients]


    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    random.seed(settings['seed'])
    torch.cuda.manual_seed_all(settings['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    experiment_name = settings['experiment_name']
    test = settings['type']
    n_epochs = settings['n_epochs']
    n_patients_UCB = settings['n_clients_UCB'] #do this later!

    k = settings['k']
   
    p2p = IFCA(patients_left, data_beats_tr, 
    data_beats_v,data_beats_t, test='IFCA')
    p2p.loop(n_epochs, experiment_name, k)