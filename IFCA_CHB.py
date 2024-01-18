### We first load the dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import h5py
import torch.nn.functional as F


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
import copy

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
        filepath = '/mimer/NOBACKUP/groups/naiss2023-22-980/arthur/'
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
                data = data.cuda()
                target = target.type(torch.LongTensor).squeeze(1).cuda()
                # Forward pass
                data = torch.reshape(data, (-1, 1, 2 * 1024))
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
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))
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
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))
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
                    data = data.cuda()
                    target = target.type(torch.LongTensor).squeeze(1).cuda()
                    # Forward pass
                    data = torch.reshape(data, (-1, 1, 2 * 1024))

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
    k = settings['k']
   
    p2p = IFCA(patients_left, settings['n_clients_UCB'], 
    test='IFCA')
    p2p.loop(n_epochs, experiment_name, k)