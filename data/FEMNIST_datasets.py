import torch
from torch.utils.data import Dataset
import numpy as np
import random

class FEMNIST_dataset(Dataset):
    def __init__(self, dataset, clients, client=0):
        super(Dataset, self).__init__()

        #assert ~isinstance(client, int) and client != 'all', "input should be integer specifying client or 'all'."
        if client == 'all':
            print('client is all')
            self.y_client = dataset[clients[0]]['y']
            self.x_client = np.array(dataset[clients[0]]['x'])
            self.clients = clients[1:]
            for client in self.clients:
                self.y_client = self.y_client + dataset[client]['y']
                self.x_client = np.concatenate((self.x_client, np.array(dataset[client]['x'])))
        if isinstance(client, int):
            self.x_client = dataset[clients[client]]['x']
            self.y_client = dataset[clients[client]]['y']

    def __len__(self):
        return len(self.y_client)

    def __getitem__(self, index):
        x = torch.from_numpy(np.array(self.x_client[index]))
        x = torch.reshape(x, (-1, 28, 28))
        return x.double(), torch.from_numpy(np.array(self.y_client[index])).double()


class Partition_MNIST_NIID(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_classes = np.unique(x)

    def divide_MNIST_per_class(self):
        self.class_division = {}  # records the indices of the 10 classes
        self.class_sample_size = []  # records the sample size of each class
        for i in range(10):
            list_name = 'MNIST_y_' + str(i)
            self.class_division[list_name] = [index for index, p in enumerate(self.x) if self.y[index] == i]
            self.class_sample_size.append(len(self.class_division[list_name]))

    def given_n_choose_k_n_client_times(self, n, k, n_clients):
        # finds n_client different partitions
        random.seed(1)
        #samples = set() use a set if we want all combinations to be different
        samples = []
        #tries = 0
        A = list(range(n))
        while len(samples) < n_clients:
            #samples.add(tuple(sorted(random.sample(A, k))))
            samples.append(random.sample(A,k))
            #tries += 1
        #self.sample_array = np.array(list(samples))
        self.sample_array = np.array(samples)
        print(self.sample_array)
        #self.sample_array = np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
        #print(self.sample_array)
        #print('sample array: ',self.sample_array)

    def find_n_samples_per_class_per_client(self):
        # count frequency of each class
        bins = [0,1,2,3,4,5,6,7,8,9,10]
        hist, _ = np.histogram(self.sample_array,bins=bins)
        epsilon = 0.0001
        self.n_samples_per_class_per_client = [int(x) for x in
                                               np.floor(np.array(self.class_sample_size) / (hist + epsilon))]
        #print(self.n_samples_per_class_per_client)
    def create_partition(self, n, k, n_clients):
        self.divide_MNIST_per_class()
        self.given_n_choose_k_n_client_times(n, k, n_clients)
        self.find_n_samples_per_class_per_client()

        dic = self.class_division
        self.partition = {}
        for i in range(n_clients):
            y_client = []
            client_id = 'client_' + str(i)
            samples = self.sample_array[i, :]
            # print(samples)
            for sample in samples:
                list_name = 'MNIST_y_' + str(sample)
                y_client.append(dic[list_name][:self.n_samples_per_class_per_client[sample]])
                dic[list_name] = dic[list_name][self.n_samples_per_class_per_client[sample]:]
            self.partition[client_id] = [item for sublist in y_client for item in sublist]
        return self.partition

    def create_partition_test(self, sample_array):
        self.divide_MNIST_per_class()
        self.sample_array = sample_array
        #print(self.sample_array)
        n_clients = len(sample_array)
        self.find_n_samples_per_class_per_client()

        dic = self.class_division
        self.partition = {}
        for i in range(n_clients):
            y_client = []
            client_id = 'client_' + str(i)
            samples = self.sample_array[i, :]
            # print(samples)
            for sample in samples:
                list_name = 'MNIST_y_' + str(sample)
                y_client.append(dic[list_name][:self.n_samples_per_class_per_client[sample]])
                dic[list_name] = dic[list_name][self.n_samples_per_class_per_client[sample]:]
            self.partition[client_id] = [item for sublist in y_client for item in sublist]
        return self.partition


class MNIST_NIID_dataset(Dataset):
    def __init__(self, x, y, partition, client=0):
        super(Dataset, self).__init__()
        self.n_classes = np.unique(y)
        self.x = x
        self.y = y
        self.partition = partition
        self.client = client
        self.extract_x_y_from_partition()

    def extract_x_y_from_partition(self):
        list_name = 'client_' + str(self.client)
        self.y_client = self.y[self.partition[list_name]]
        self.x_client = self.x[self.partition[list_name]]

    def __len__(self):
        return len(self.y_client)

    def __getitem__(self, index):
        x = torch.from_numpy(np.array(self.x_client[index]))
        x = torch.reshape(x, (-1, 28, 28)).double()
        x = (x-torch.mean(x))/torch.std(x)
        return x, torch.from_numpy(np.array(self.y_client[index])).double()