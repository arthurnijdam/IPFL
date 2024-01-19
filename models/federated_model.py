import torch
from torch.utils.data import Dataset
import numpy as np


class FEMNIST_dataset(Dataset):
    def __init__(self, dataset, clients, client=0):
        super(Dataset, self).__init__()

        assert ~isinstance(client, int) and client != 'all', "input should be integer specifying client or 'all'."
        if client == 'all':
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
