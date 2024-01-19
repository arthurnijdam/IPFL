
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import default_rng
import collections
from torch.utils.data import DataLoader,ConcatDataset
from data.FEMNIST_datasets import FEMNIST_dataset, Partition_MNIST_NIID, MNIST_NIID_dataset
from tensorflow import keras
from torch.utils.data import Subset
from utils import read_data
import numpy as np
import random
import os
import copy
from torchmetrics import Accuracy

def calc_confusion_matrix(targets,outputs,print_result=False):
    confusion_matrix = np.zeros((10, 10))
    targets_array = np.zeros((10, 1))

    for ii, target in enumerate(targets):
        output = outputs[ii]
        for i, digit in enumerate(target):
            # print(target[i])
            # print(output[i])
            output_labels = np.argmax(output[i], axis=-1)
            # print(np.shape(output))
            confusion_matrix[target[i], output_labels] += 1
            targets_array[target[i]] += 1
    if print_result==True:
        np.set_printoptions(suppress=True)
        #print(confusion_matrix)
        print(targets_array)
    return confusion_matrix

def accuracy(confusion_matrix,print_result=True,return_acc=False):
    correct = np.trace(confusion_matrix)
    total = np.sum(confusion_matrix)
    if print_result == True:
        #print(correct)
        #print(total)
        print('accuracy = ',correct/total*100,'%')
    if return_acc == True:
        return correct/total*100

class Net(nn.Module):
    def __init__(self,dataset,out=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        if dataset == 'FEMNIST':
            self.fc2 = nn.Linear(100, 62)
        if dataset == 'MNIST_niid':
            self.fc2 = nn.Linear(100,out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Local_Model(object):
    def __init__(self,device, network, total_clients = 100,
        batch_size = 16, dataset = 'FEMNIST',classes_per_user=4):
        self.device = device
        self.network = network
        self.batch_size = batch_size
        self.dataset = dataset
        self.classes_per_user = classes_per_user
        if self.dataset == 'FEMNIST':
            train_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/train'
            test_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/test'
            self.clients, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
            self.train_loader,self.test_loader = self.global_FEMNIST()
        if self.dataset == 'MNIST_niid':
            self.train_data, self.test_data = keras.datasets.mnist.load_data()
            instance = Partition_MNIST_NIID(self.train_data[0], self.train_data[1])
            self.train_partition = instance.create_partition(10, classes_per_user, total_clients)
            test_instance = Partition_MNIST_NIID(self.test_data[0], self.test_data[1])
            self.test_partition = test_instance.create_partition_test(instance.sample_array)
            #print(instance.sample_array)
            #breakpoint()

    def train_loop(self,save_dir,n_epochs=30,learning_rate=0.01,momentum=0.5):
        self.client_models = {}
        self.losses = {}
        self.test_losses = {}
        self.best_test_loss = {}
        for client in self.train_partition:
            self.losses[client] = []
            self.test_losses[client] = []
            self.best_test_loss[client] = 100000
            self.global_model = self.network.double().to(self.device)
            number = int(client[7:])
            dataset = MNIST_NIID_dataset(self.train_data[0],self.train_data[1],self.train_partition,number)
            self.train_loader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
            test_dataset = MNIST_NIID_dataset(self.test_data[0], self.test_data[1], self.test_partition, number)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

            self.local_optimizer = optim.SGD(self.global_model.parameters(), lr=learning_rate, momentum=momentum)
            for epoch in range(n_epochs):
                train_loss = self.train_one_epoch()
                test_loss = self.test_one_epoch()
                # torch.save(federated_averaging.global_model.state_dict(),os.path.join(save_dir,'model',str(i)+'_model.pt'))
                with open(os.path.join(save_dir, 'loss.txt'), 'w') as f:
                    f.write('client: '+str(number))
                    for i, _ in enumerate(self.losses[client]):
                        f.write('epoch: ' + str(
                            i) + ' train loss: ' + f"{self.losses[client][i]}" + ' test loss: ' + f"{self.test_losses[client][i]}\n")

                self.losses[client].append(train_loss)
                self.test_losses[client].append(test_loss)
                if test_loss < self.best_test_loss[client]:
                    self.client_models[client] = self.global_model.state_dict()
                    self.best_test_loss[client] = test_loss
        return self.client_models

    def evaluation_loop(self,client_model,number):
        self.global_model = client_model

        dataset = MNIST_NIID_dataset(self.train_data[0],self.train_data[1],self.train_partition,number)
        self.train_loader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        test_dataset = MNIST_NIID_dataset(self.test_data[0], self.test_data[1], self.test_partition, number)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)


        targets,outputs= self.test_one_epoch(calculate_accuracy=True)
        #breakpoint()
        return targets,outputs

    def train_one_epoch(self):
        losses = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.double().to(self.device)
            target = target.long().to(self.device)
            self.local_optimizer.zero_grad()
            output = self.global_model(data)
            loss = F.nll_loss(output, target)
            losses = torch.sum(loss)
            loss.backward()
            self.local_optimizer.step()
            print(target)
        return losses.detach().cpu().numpy() / len(self.train_loader)

    def test_one_epoch(self,calculate_accuracy=True):
        losses = 0
        targets = []
        outputs = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.double().to(self.device)
                target = target.long().to(self.device)

                output = self.global_model(data)
                loss = F.nll_loss(output, target)
                losses = torch.sum(loss)
                if calculate_accuracy == True:
                    targets.append(target.detach().cpu().numpy().ravel())
                    outputs.append(output.detach().cpu().numpy())
                print(target[0])
        if calculate_accuracy==False:
            return losses.detach().cpu().numpy() / len(self.test_loader)
        else:
            confusion_matrix = calc_confusion_matrix(targets,outputs,print_result=True)
            accuracy(confusion_matrix)
            return losses.detach().cpu().numpy() / len(self.test_loader)
            #return targets,outputs

class Global_Model(object):
    def __init__(self,device, network, n_clients = 1, total_clients = 1,
        batch_size = 16, dataset = 'FEMNIST',classes_per_user=4,same_as_federated=False):
        self.device = device
        self.global_model = network.double().to(self.device)
        self.client_ratio = n_clients/total_clients
        self.batch_size = batch_size
        self.dataset = dataset
        self.classes_per_user = classes_per_user
        self.total_clients = total_clients
        self.same_as_federated = same_as_federated
        if self.dataset == 'FEMNIST':
            train_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/train'
            test_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/test'
            self.clients, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
            self.train_loader,self.test_loader = self.global_FEMNIST()

        if self.dataset == 'MNIST_niid':
            self.train_loader,self.test_loader = self.global_MNIST_NIID()

        if self.same_as_federated == True:
            self.train_data, self.test_data = keras.datasets.mnist.load_data()
            instance = Partition_MNIST_NIID(self.train_data[0], self.train_data[1])
            self.train_partition = instance.create_partition(10, classes_per_user, total_clients)
            test_instance = Partition_MNIST_NIID(self.test_data[0], self.test_data[1])
            self.test_partition = test_instance.create_partition_test(instance.sample_array)
            self.n_clients = n_clients
            self.total_clients = total_clients
            #self.train_loader,self.test_loader = self.global_MNIST_federated()
        random.seed(1)
    def select_clients(self, n_clients=1, total_clients=1,seed=1):
        #rng = default_rng()
        random.seed(seed)
        l = [i for i in range(total_clients)]

        #np.random.seed(0)
        self.selected_clients = random.sample(l,k=n_clients) #rng.choice(total_clients, size=n_clients, replace=False)
        print(self.selected_clients)
        return self.selected_clients


    def global_FEMNIST(self):
        data_train = FEMNIST_dataset(self.train_data, self.clients, 'all')
        data_train = Subset(data_train,indices=range(int(np.ceil(self.client_ratio*len(data_train)))))
        train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        data_test = FEMNIST_dataset(self.test_data, self.clients, 'all')
        data_test = Subset(data_test, indices=range(int(np.ceil(self.client_ratio * len(data_test)))))
        test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def global_MNIST_federated(self,seed=1):
        self.selected_clients = self.select_clients(self.n_clients, self.total_clients,seed)
        dataset_train = []
        dataset_test = []
        for i in self.selected_clients:
            for partition in ["train", "test"]:
                if partition == "train":
                    data = self.train_data
                    if self.dataset == 'FEMNIST':
                        dataset_train.append(FEMNIST_dataset(data, self.clients, int(i)))
                    if self.dataset == 'MNIST_niid':
                        if partition == "train":
                            self.partition = self.train_partition
                        else:
                            self.partition = self.test_partition
                        dataset_train.append(MNIST_NIID_dataset(data[0], data[1], self.partition,i))
                if partition == "test":
                    data = self.test_data
                    if self.dataset == 'FEMNIST':
                        dataset_test.append(FEMNIST_dataset(data, self.clients, int(i)))
                    if self.dataset == 'MNIST_niid':
                        if partition == "train":
                            self.partition = self.train_partition
                        else:
                            self.partition = self.test_partition
                        dataset_test.append(MNIST_NIID_dataset(data[0], data[1], self.partition,i))
        dataset_train= ConcatDataset(dataset_train)
        train_loader = DataLoader(dataset_train,batch_size=self.batch_size,shuffle=False)
        dataset_test = ConcatDataset(dataset_test)
        test_loader = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def global_MNIST_NIID(self):
        data_train = torchvision.datasets.MNIST('/mimer/NOBACKUP/groups/snic2022-22-122/Arthur/code/Federated_Averaging/files/',
                                                                              train=True, download=True,
                                                                              transform=torchvision.transforms.Compose([
                                                                                  torchvision.transforms.ToTensor(),
                                                                                  torchvision.transforms.Normalize(
                                                                                      (0.1307,), (0.3081,))
                                                                              ]))
        print(int(np.ceil(self.client_ratio * len(data_train))))
        data_train = Subset(data_train, indices=range(int(np.ceil(self.client_ratio * len(data_train)))))
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        data_test = torchvision.datasets.MNIST('/mimer/NOBACKUP/groups/snic2022-22-122/Arthur/code/Federated_Averaging/files/',
                                                                             train=False, download=True,
                                                                             transform=torchvision.transforms.Compose([
                                                                                 torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.Normalize(
                                                                                     (0.1307,), (0.3081,))
                                                                             ]))
        data_test = Subset(data_test, indices=range(int(np.ceil(self.client_ratio * len(data_test)))))
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def train_one_epoch(self,learning_rate=0.01, momentum=0.5,epoch=0):
        if self.same_as_federated == True:
            self.train_loader,self.test_loader = self.global_MNIST_federated(seed=epoch)
        losses = 0
        local_optimizer = optim.SGD(self.global_model.parameters(), lr=learning_rate, momentum=momentum)
        for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.double().to(self.device)
                target = target.long().to(self.device)

                local_optimizer.zero_grad()
                output = self.global_model(data)
                loss = F.nll_loss(output, target)
                losses = torch.sum(loss)
                loss.backward()
                local_optimizer.step()
        return losses.detach().cpu().numpy() / len(self.train_loader)

    def evaluate_one_epoch(self,epoch=0,calc_accuracy=False):
        if self.same_as_federated == True:
            self.train_loader,self.test_loader = self.global_MNIST_federated(epoch)
        losses = 0
        targets = []
        outputs = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.double().to(self.device)
                target = target.long().to(self.device)

                output = self.global_model(data)
                losses += F.nll_loss(output, target)
                if calc_accuracy == True:
                    targets.append(target.detach().cpu().numpy().ravel())
                    outputs.append(output.detach().cpu().numpy())
        if calc_accuracy == False:
            return losses.detach().cpu().numpy() / len(self.test_loader)  # targets, outputs#
        else:
            return targets, outputs
        #return losses.detach().cpu().numpy() / len(self.test_loader)


class Federated_Averaging(object):
    def __init__(self, device, network, n_clients=1, total_clients=1, classes_per_user=4,
                 batch_size=16, dataset='FEMNIST'):
        self.device = device
        self.global_model = network.double().to(self.device)
        self.select_clients(n_clients, total_clients)
        self.n_clients = n_clients
        self.total_clients = total_clients
        self.batch_size = batch_size
        self.dataset = dataset
        if self.dataset == 'FEMNIST':
            train_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/train'
            test_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/test'
            self.clients, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
            #print(len(self.clients))
            #breakpoint()
        if self.dataset == 'MNIST_niid':
            self.train_data, self.test_data = keras.datasets.mnist.load_data()
            instance = Partition_MNIST_NIID(self.train_data[0], self.train_data[1])
            self.train_partition = instance.create_partition(10, classes_per_user, total_clients)
            test_instance = Partition_MNIST_NIID(self.test_data[0], self.test_data[1])
            self.test_partition = test_instance.create_partition_test(instance.sample_array)
            #breakpoint()
        random.seed(1)
    def select_clients(self, n_clients=1, total_clients=1,seed=1):
        # rng = default_rng()
        random.seed(seed)
        l = [i for i in range(total_clients)]


        self.selected_clients = random.sample(l,
                                               k=n_clients)  # rng.choice(total_clients, size=n_clients, replace=False)
        #print(self.selected_clients)
        return self.selected_clients

    def create_dataloaders(self, learning_rate=0.01, momentum=0.5):
        self.dataloader_dict = {}
        self.n_samples_train = []
        self.n_samples_test = []

        for i in self.selected_clients:
            for partition in ["train", "test"]:
                if partition == "train":
                    data = self.train_data
                if partition == "test":
                    data = self.test_data
                data_name = "data_" + str(i) + partition
                if self.dataset == 'FEMNIST':
                    local_data = FEMNIST_dataset(data, self.clients, int(i))
                    train_loader = DataLoader(local_data, batch_size=self.batch_size, shuffle=True)
                if self.dataset == 'MNIST_niid':
                    if partition == "train":
                        self.partition = self.train_partition
                    else:
                        self.partition = self.test_partition
                    client_name = 'client_' + str(i)
                    #print(self.partition[client_name])
                    #breakpoint()
                    local_data = MNIST_NIID_dataset(data[0], data[1], self.partition,i)
                    #print(len(local_data))
                    # print(np.shape(data[self.partition[client_name]]))
                    # print(np.shape(data[1][self.partition[client_name]]))
                    # local_data = (data[1][self.partition[client_name]],data[0][self.partition[client_name]])
                    train_loader = DataLoader(local_data, batch_size=self.batch_size, shuffle=True)
                    # test_loader = DataLoader(self.test_data[0][self.test_partition[client_name]],self.test_data[1][self.test_partition[client_name]],batch_size=self.batch_size,shuffle=True)
                if partition == "train":
                    self.n_samples_train.append(len(local_data))
                if partition == "test":
                    self.n_samples_test.append(len(local_data))

                self.dataloader_dict.update({data_name: train_loader})
        return self.dataloader_dict, self.n_samples_train

    def compute_weights(self):
        self.train_weights = [ele / sum(self.n_samples_train) for ele in self.n_samples_train]
        self.test_weights = [ele / sum(self.n_samples_test) for ele in self.n_samples_test]
        return self.train_weights, self.test_weights

    def create_empty_model(self):
        self.global_model_intermediate = collections.OrderedDict(
            [(name, torch.zeros(param.shape).to(self.device)) for name, param in
             self.global_model.state_dict().items()])

    def setup(self,epoch=0):
        self.select_clients(self.n_clients,self.total_clients,seed=epoch)
        self.create_empty_model()
        self.create_dataloaders()
        self.compute_weights()

    def aggregate_local_model(self, idx, local_model):
        weight = self.train_weights[idx]
        self.global_model_intermediate = collections.OrderedDict(
            [(name, weight * param + self.global_model_intermediate[name]) for name, param in
             local_model.state_dict().items()])

    def update_global_model(self):
        for name, param in self.global_model.named_parameters():
            param.data = self.global_model_intermediate[name] #/ len(self.selected_clients)

    def train_one_epoch(self, local_iterations=1, learning_rate=0.01, momentum=0.5):
        # print('selected clients: ',self.selected_clients)
        losses = 0
        for idx, i in enumerate(self.selected_clients):
            dataloader_name = "data_" + str(i) + "train"
            local_model = self.global_model
            local_optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=momentum)
            for _ in range(local_iterations):
                for batch_idx, (data, target) in enumerate(self.dataloader_dict[dataloader_name]):
                    data = data.double().to(self.device)
                    target = target.long().to(self.device)

                    local_optimizer.zero_grad()
                    output = local_model(data)
                    loss = F.nll_loss(output, target)
                    losses += torch.sum(loss)
                    loss.backward()
                    local_optimizer.step()
            # print('client ',str(i),' is done training')
            self.aggregate_local_model(idx, local_model)
            # print('value after:',global_model_intermediate['conv1.weight'][0,0,0,0])
        self.update_global_model()
        return losses.detach().cpu().numpy() #/ len(self.dataloader_dict[dataloader_name])

    def evaluate_one_epoch(self,calc_accuracy=False):
        targets = []
        outputs = []
        losses = 0
        with torch.no_grad():
            for idx, i in enumerate(self.selected_clients):
                dataloader_name = "data_" + str(i) + "test"
                for batch_idx, (data, target) in enumerate(self.dataloader_dict[dataloader_name]):
                    data = data.double().to(self.device)
                    target = target.long().to(self.device)

                    output = self.global_model(data)
                    losses += F.nll_loss(output, target)

                    if calc_accuracy == True:
                        targets.append(target.detach().cpu().numpy().ravel())
                        outputs.append(output.detach().cpu().numpy())
        if calc_accuracy == False:
            return losses.detach().cpu().numpy() #/ len(self.dataloader_dict[dataloader_name]) #targets, outputs#
        else:
            return targets, outputs

class AFPL(object):
    def __init__(self, device, network, n_clients=1, total_clients=1, classes_per_user=4,
                 batch_size=16, dataset='FEMNIST'):
        self.device = device
        self.global_model = network.double().to(self.device)
        self.select_clients(n_clients, total_clients)
        self.n_clients = n_clients
        self.total_clients = total_clients
        self.batch_size = batch_size
        self.dataset = dataset
        if self.dataset == 'FEMNIST':
            train_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/train'
            test_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/test'
            self.clients, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
        if self.dataset == 'MNIST_niid':
            self.train_data, self.test_data = keras.datasets.mnist.load_data()
            instance = Partition_MNIST_NIID(self.train_data[0], self.train_data[1])
            self.train_partition = instance.create_partition(10, classes_per_user, total_clients)
            test_instance = Partition_MNIST_NIID(self.test_data[0], self.test_data[1])
            self.test_partition = test_instance.create_partition_test(instance.sample_array)
            # breakpoint()
        random.seed(1)
        #self.alpha = 1
        self.initialize_local_models()
    def select_clients(self, n_clients=1, total_clients=1, seed=1):
        # rng = default_rng()
        random.seed(seed)
        l = [i for i in range(total_clients)]

        self.selected_clients = random.sample(l,
                                              k=n_clients)  # rng.choice(total_clients, size=n_clients, replace=False)
        # print(self.selected_clients)
        return self.selected_clients #adjusted for flexible alpha implementation

    def create_dataloaders(self, learning_rate=0.01, momentum=0.5):
        self.dataloader_dict = {}
        self.n_samples_train = []
        self.n_samples_test = []

        for i in self.selected_clients:
            for partition in ["train", "test"]:
                if partition == "train":
                    data = self.train_data
                if partition == "test":
                    data = self.test_data
                data_name = "data_" + str(i) + partition
                if self.dataset == 'FEMNIST':
                    local_data = FEMNIST_dataset(data, self.clients, int(i))
                    train_loader = DataLoader(local_data, batch_size=self.batch_size, shuffle=True)
                if self.dataset == 'MNIST_niid':
                    if partition == "train":
                        self.partition = self.train_partition
                    else:
                        self.partition = self.test_partition
                    client_name = 'client_' + str(i)
                    # print(self.partition[client_name])
                    # breakpoint()
                    local_data = MNIST_NIID_dataset(data[0], data[1], self.partition, i)
                    # print(len(local_data))
                    # print(np.shape(data[self.partition[client_name]]))
                    # print(np.shape(data[1][self.partition[client_name]]))
                    # local_data = (data[1][self.partition[client_name]],data[0][self.partition[client_name]])
                    train_loader = DataLoader(local_data, batch_size=self.batch_size, shuffle=True)
                    # test_loader = DataLoader(self.test_data[0][self.test_partition[client_name]],self.test_data[1][self.test_partition[client_name]],batch_size=self.batch_size,shuffle=True)
                if partition == "train":
                    self.n_samples_train.append(len(local_data))
                if partition == "test":
                    self.n_samples_test.append(len(local_data))

                self.dataloader_dict.update({data_name: train_loader})
        return self.dataloader_dict, self.n_samples_train

    def compute_weights(self):
        self.train_weights = [ele / sum(self.n_samples_train) for ele in self.n_samples_train]
        self.test_weights = [ele / sum(self.n_samples_test) for ele in self.n_samples_test]
        ### Change back!!!!
        #self.train_weights = [1 / self.n_clients for ele in self.n_samples_train]
        #self.test_weights = [1 / self.n_clients for ele in self.n_samples_test]
        return self.train_weights, self.test_weights

    def create_empty_model(self):
        self.global_model_intermediate = collections.OrderedDict(
            [(name, torch.zeros(param.shape).to(self.device)) for name, param in
             self.global_model.state_dict().items()])
        #self.global_grad_intermediate = collections.OrderedDict( #added for alternative alpha implementation
         #   [(name, torch.zeros(param.shape).to(self.device)) for name, param in
          #   self.global_model.state_dict().items()])

    def initialize_local_models(self, learning_rate=0.01, momentum=0.5):
        self.client_models = {}
        self.local_optimizers = {}
        self.alphas = {}
        for idx in range(self.total_clients):
            self.client_models[idx] = self.global_model
            self.local_optimizers[idx] = optim.SGD(self.client_models[idx].parameters(), lr=learning_rate, momentum=momentum)
            self.alphas[idx] = 0.5

    def setup(self, epoch=0):
        self.select_clients(self.n_clients, self.total_clients, seed=epoch)
        self.create_empty_model()
        self.create_dataloaders()
        self.compute_weights()
        print('selected clients: ',self.selected_clients)

    def aggregate_local_model(self, idx, local_model,grad=True):
        weight = self.train_weights[idx]
        self.global_model_intermediate = collections.OrderedDict(
            [(name, weight * param + self.global_model_intermediate[name]) for name, param in
             local_model.state_dict().items()])
        #for param in local_model.parameters():
         #   print('name')
          #  print(param)
           # print(param.grad.data)
            #breakpoint()

        #self.global_grad_intermediate = collections.OrderedDict( #added for alternative alpha implementation
        #    [(name, weight * param.grad.data + self.global_grad_intermediate[name]) for name, param in
        #     local_model.named_parameters()])


    def update_global_model(self):
        for name, param in self.global_model.named_parameters():
            param.data = self.global_model_intermediate[name]
            #print(self.global_grad_intermediate[name])

            #param.grad = self.global_grad_intermediate[name] # added for alternative alpha implementation

    def update_local_model(self,idx,global_model,local_model): ### Adjust this back to global_model to follow the official implementation
        for local_param,param in zip(local_model.parameters(),global_model.parameters()):
            local_param.data = param.data*(1-self.alpha) + self.alpha*local_param.data
        self.client_models[idx] = copy.deepcopy(local_model)

    def alpha_update(self,alpha,local_model,global_model,idx,learning_rate=0.1):
        grad_alpha = 0
        for l_params, p_params in zip(global_model.parameters(), local_model.parameters()): ### Adjust this back to global_model to follow the official implementation
            dif = p_params.data - l_params.data
            #print(dif)
            grad = alpha * p_params.grad.data + (1 - alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * alpha
        alpha = alpha - learning_rate * grad_alpha
        self.alphas[idx] = np.clip(alpha.item(), 0.0, 1.0)
        #breakpoint()

    def train_one_epoch(self, epoch,local_iterations=1, learning_rate=0.01, momentum=0.5):
        # print('selected clients: ',self.selected_clients)
        losses = 0
        losses_local = 0
        losses_global = 0
        for idx, i in enumerate(self.selected_clients):
            dataloader_name = "data_" + str(i) + "train"
            global_model = copy.deepcopy(self.global_model)
            local_model = copy.deepcopy(self.client_models[i])
            #if local_model == global_model:
            #    print('local model and global model are the same?')
            #else:
             #   print('NOT THE SAME')
            self.alpha = self.alphas[i]
            local_optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=momentum)
            global_optimizer = optim.SGD(global_model.parameters(), lr=learning_rate, momentum=momentum)
            for _ in range(local_iterations):
                for batch_idx, (data, target) in enumerate(self.dataloader_dict[dataloader_name]):
                    data = data.double().to(self.device)
                    target = target.long().to(self.device)
                    local_optimizer.zero_grad()
                    global_optimizer.zero_grad()
                    output_global = global_model(data)
                    loss_global = F.nll_loss(output_global, target)
                    output_local = local_model(data)
                    loss_local= F.nll_loss(output_local, target)
                    losses += torch.sum((1-self.alpha)*loss_global+self.alpha*loss_local) #/self.batch_size
                    losses_local += torch.sum(loss_local)
                    losses_global += torch.sum(loss_global)

                    loss_local.backward()
                    loss_global.backward()
                    global_optimizer.step()
                    local_optimizer.step()

                #losses = losses / batch_idx
                #losses_local = losses_local / batch_idx
                #losses_global = losses_global / batch_idx
            # print('client ',str(i),' is done training')
            #print('client',str(i))
            #print(target.detach().cpu().numpy())
            self.aggregate_local_model(idx, global_model)
            if epoch > 0: #only in the case of the adjusted alpha!
                self.alpha_update(self.alpha,local_model,global_model,i)
            # print('value after:',global_model_intermediate['conv1.weight'][0,0,0,0])
            #print(idx)
            #print(self.alpha)
            self.update_local_model(i,global_model,local_model)
            #if self.client_models[1] == self.client_models[0]:
             #   print('the two client models are identical!'
             #   )
            #else:
            #    print('the two client models are not identical!')
        self.update_global_model()  # not 100% sure on the placement, but since it says 'parallell' I think this is right
        #breakpoint()
        return self.alphas[0],self.alphas[1],self.alphas[2] #losses_global.detach().cpu().numpy()/self.n_clients,losses_local.detach().cpu().numpy()/self.n_clients ,self.alpha #,losses_global.detach().cpu().numpy()/self.n_clients,self.alpha#/ len(self.dataloader_dict[dataloader_name])


    def evaluate_one_epoch(self, calc_accuracy=True):
        targets = []
        outputs = []
        losses = 0
        with torch.no_grad():
            for idx, i in enumerate(self.selected_clients):
                local_model = self.client_models[i]
                dataloader_name = "data_" + str(i) + "test"
                for batch_idx, (data, target) in enumerate(self.dataloader_dict[dataloader_name]):
                    data = data.double().to(self.device)
                    target = target.long().to(self.device)

                    output = local_model(data) #self.global_model(data)
                    losses += torch.sum(F.nll_loss(output, target))

                    if calc_accuracy == True:
                        targets.append(target.detach().cpu().numpy().ravel())
                        outputs.append(output.detach().cpu().numpy())
        if calc_accuracy == False:
            return losses.detach().cpu().numpy() / self.n_clients #self.dataloader_dict[dataloader_name])  # targets, outputs#
        else:
            confusion_matrix = calc_confusion_matrix(targets, outputs, print_result=False)
            self.accuracy = accuracy(confusion_matrix,print_result=False,return_acc=True)

            return losses.detach().cpu().numpy() / self.n_clients
            #return targets, outputs



