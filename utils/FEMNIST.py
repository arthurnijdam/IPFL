import os
import json
from string import ascii_lowercase as alc
import matplotlib.pyplot as plt
import numpy as np
# Use this function to read the FEMNIST data format (taken from FedRep directory)
def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir ,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir ,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data

def label_mapping_FEMNIST(i):
    if i <= 9:
        label_mapping = str(i)
    if i >9 and i<9+26+1:
        label_mapping = alc[i-10]
    if i >= 9+26+1:
        label_mapping = alc[i-36].upper()
    return label_mapping


def plot_random_sample_mnist(dataset, clients):
    client = np.random.randint(len(dataset))
    data_client = dataset[clients[client]]
    x_client = data_client['x']
    y_client = data_client['y']
    index = np.random.randint(np.shape(x_client)[0])
    image_1 = np.reshape(x_client[index], (28, 28))
    plt.imshow(image_1, cmap='gray')
    plt.title(label_mapping_FEMNIST(y_client[index]))


def plot_histogram_classes_FEMNIST(dataset, client):
    y_client = dataset[clients[client]]['y']
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    counts, bins, patches = ax.hist(y_client, bins=62)
    # plt.hist(y_client,bins=62)
    axis_labels = [str(i) for i in range(10)] + [i for i in alc] + [i.upper() for i in alc]
    ax.set_xticks(np.arange(62), axis_labels)
    ax.set_xlabel('classes')
    ax.set_ylabel('occurrence')


def plot_histogram_classes_FEMNIST_whole(dataset):
    y_client = []
    for client in dataset:
        y_client = y_client + dataset[client]['y']
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    counts, bins, patches = ax.hist(y_client, bins=62)
    # plt.hist(y_client,bins=62)
    axis_labels = [str(i) for i in range(10)] + [i for i in alc] + [i.upper() for i in alc]
    ax.set_xticks(np.arange(62), axis_labels)
    ax.set_xlabel('classes')
    ax.set_ylabel('occurrence')

if __name__=="__main__":
    train_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/train'
    test_data_dir = '/mimer/NOBACKUP/groups/snic2022-22-122/arthur/leaf-master/data/femnist/data/test'
    os.path.isdir(train_data_dir)  # check if the training directory exists
    print(os.path.isdir(train_data_dir))
    clients, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    