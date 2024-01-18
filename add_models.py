import yaml
import shutil
from models import Federated_Averaging,Global_Model,Local_Model,Net,AFPL
import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn

def flatten(source):
    list1 = []
    for value in source:
        #  print(value.flatten().shape)
        list1.append(value.flatten())
    # print(len(list1))
    return torch.cat(list1)  # deepcopy(value).flatten() for value in source])

def plot_loss_curves(train_losses,alpha):
    train_losses = np.array(train_losses)
    alpha = np.array(alpha)
    experiment_name = "test"
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join('checkpoints', experiment_name, 'loss_curve_alpha.png'))
    #plt.show()
    plt.figure()
    plt.plot(alpha, label="alpha")
    # plt.plot(test_losses, label="test loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join('checkpoints', experiment_name, 'progression_alpha.png'))
    #plt.show()

class Net3(nn.Module):
    def __init__(self,dataset,alphas,out=10):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1.requires_grad = False
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2.requires_grad = False
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.fc1.requires_grad = False
        if dataset == 'FEMNIST':
            self.fc2 = nn.Linear(100, 62)
        if dataset == 'MNIST_niid':
            self.fc2 = nn.Linear(100,out)
        self.fc2.requires_grad = False
        self.alphas = nn.Parameter(alphas)

    def forward(self, x,client_model):
        list1 = ['conv1','conv2','fc1','fc2']
        x = F.relu(F.max_pool2d(combine_layer(self.conv1,client_model,self.alphas,x,list1,0), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(combine_layer(self.conv2,client_model,self.alphas,x,list1,1)), 2))
        x = x.view(-1, 320)
        x = F.relu(combine_layer(self.fc1,client_model,self.alphas,x,list1,2))
        x = F.dropout(x, training=self.training)

        x = combine_layer(self.fc2,client_model,self.alphas,x,list1,3)

        return F.log_softmax(x)

def combine_layer(layer1,client_model,alphas,x,layer_list,layer_idx):
    x1 = layer1(x)
    layer2 = getattr(client_model,layer_list[layer_idx])
    x2 = layer2(x) # client_model.state_dict()[layer_number](x)
    x =  (1-alphas[0])*x1 + alphas[0]*x2
    return x

class Net4(nn.Module):
    def __init__(self,dataset,alphas,out=10):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1.requires_grad = False
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2.requires_grad = False
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.fc1.requires_grad = False
        if dataset == 'FEMNIST':
            self.fc2 = nn.Linear(100, 62)
        if dataset == 'MNIST_niid':
            self.fc2 = nn.Linear(100,out)
        self.fc2.requires_grad = False
        self.alphas = nn.Parameter(alphas)

    def forward(self, x,i=None,client_models=None,client_numbers=[0]):

        if client_models != None:

            list1 = ['conv1','conv2','fc1','fc2']
            x = F.relu(F.max_pool2d(combine_layers(self.conv1,client_models,client_numbers,self.alphas,x,list1,0), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(combine_layers(self.conv2,client_models,client_numbers,self.alphas,x,list1,1)), 2))
            x = x.view(-1, 320)
            x = F.relu(combine_layers(self.fc1,client_models,client_numbers,self.alphas,x,list1,2))
            x = F.dropout(x, training=self.training)

            x = combine_layers(self.fc2,client_models,client_numbers,self.alphas,x,list1,3)


        else:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)


        return F.log_softmax(x)

def combine_layers(layer1,client_models,client_numbers,alphas,x,layer_list,layer_idx):

    alpha_remainder = 1
    for ii in client_numbers:
        layer2 = getattr(client_models[str(ii)],layer_list[layer_idx])

        x2 = layer2(x) # client_model.state_dict()[layer_number](x)
        x_int = alphas[ii]*x2

        alpha_remainder = alpha_remainder - alphas[ii]

    x1 = layer1(x)
    x =  x_int + alpha_remainder * x1  #(1-alphas[0])*x1 + alphas[0]*x2
    return x





class P2P_AFPL():
    def __init__(self, total_clients, train_data, train_partition, test_data, test_partition):
        alphas = torch.ones(total_clients,1) * (1 / total_clients)
        self.network = Net4('MNIST_niid',alphas)
        self.total_clients = total_clients
        self.client_models = {}
        self.optimizers = {}
        self.dataloaders = {}
        self.dataloaders_test = {}
        for i in range(total_clients):
            self.client_models[str(i)] = copy.deepcopy(self.network).double().cuda()
            self.optimizers[str(i)] = torch.optim.SGD(self.client_models[str(i)].parameters(), lr=0.01, momentum=0.5)
            dataset_train = MNIST_NIID_dataset(train_data[0], train_data[1], train_partition, i)  # i)
            dataset_test = MNIST_NIID_dataset(train_data[0], train_data[1], train_partition, 2)  # i)
            self.dataloaders[str(i)] = DataLoader(dataset_train, batch_size=16, shuffle=True)
            #dataset_test = MNIST_NIID_dataset(test_data[0], test_data[1], test_partition, i)  # i)
            self.dataloaders_test[str(i)] = DataLoader(dataset_test, batch_size=16, shuffle=True)


    def create_graph(self, selected_clients):
        graph = []
        for index, sample in enumerate(selected_clients):
            other_clients = copy.deepcopy(selected_clients)
            other_clients.remove(sample)
            l = [i for i in other_clients]
            graph.append(random.sample(l, k=2))

        return graph

    def select_clients(self, n_clients=1, total_clients=1, seed=1):
        random.seed(seed)
        l = [i for i in range(total_clients)]
        self.selected_clients = random.sample(l, k=n_clients)
        return self.selected_clients

    def update_local_models(self, selected_clients):

        self.dw = {}
        for idx, i in enumerate(selected_clients):

            dataloader = self.dataloaders[str(i)]

            #dataset_train = MNIST_NIID_dataset(train_data[0], train_data[1], train_partition, i)  # i)
            #self.dataloaders[str(i)] = DataLoader(dataset_train, batch_size=16, shuffle=False)
            #dataset_test = MNIST_NIID_dataset(test_data[0], test_data[1], test_partition, i)  # i)
            #self.dataloaders_test[str(i)] = DataLoader(dataset_test, batch_size=16, shuffle=False)

            # Ensure that everything is trainable except for alpha
            for name, local_param in self.client_models[str(i)].named_parameters():
                if name == 'alphas':
                    local_param.requires_grad = False
                    #print('before: ',local_param)
                    #local_param.data = torch.clamp(local_param.data,0,1) #torch.nn.functional.softmax(local_param) #clamp(local_param,0,1)
                    #print('after: ', local_param)
                    #self.client_models[str(i)].alphas[5] = torch.sigmoid(self.client_models[str(i)].alphas[5])
                    # self.client_models[str(i)].alphas[5].requires_grad = True

                else:
                    local_param.requires_grad = True

            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target = target.long().cuda()

                self.optimizers[str(i)].zero_grad()
                output = self.client_models[str(i)](data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizers[str(i)].step()

    def evaluate(self,i,model):
        dataloader = self.dataloaders_test[str(i)]
        if model == 'client_model':
            local_model = self.client_models[str(i)]
        else:
            local_model = model

        loss_test = 0
        local_model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target = target.long().cuda()

                output = local_model(data)
                loss_test += F.nll_loss(output, target)
        return loss_test

    def init_alpha(self,i,selected_clients):


        # Ensure that only alpha is trainable now
        for name, local_param in self.client_models[str(i)].named_parameters():
            if name != 'alphas':
                local_param.requires_grad = False
            else:
                local_param.requires_grad = True

        self.client_models[str(i)].eval()

        # initiate optimizer
        #optimizer_alpha = torch.optim.SGD(self.client_models[str(i)].parameters(), lr=0.001)
        optimizer_alpha = torch.optim.Adam(self.client_models[str(i)].parameters(), lr=0.01)
        # make a dict with the models of the other clients
        client_model_dict = {}
        d = [iii for iii in selected_clients if iii != i]
        #selected_clients.remove(i)
        for ii in d:
            client_model_dict[str(ii)] = self.client_models[str(ii)]

        return client_model_dict, optimizer_alpha

    def combine_models(self,i,client_numbers):
        zero_copy = copy.deepcopy(self.client_models[str(i)]) # This is used to collect the model in
        j =0
        client_numbers_plus_client = client_numbers + [i] # This is more efficient
        alphas = zero_copy.alphas.detach()
        alphas[i] = 1 - torch.sum(
            torch.tensor([iii for idx, iii in enumerate(alphas) if idx != i and idx in client_numbers]))
        # It's not possible to set the value of self.alphas[i], so instead we determine it manually here

        for ii in client_numbers_plus_client:

            for (name, param),(name2,param2) in zip(zero_copy.named_parameters(),self.client_models[str(ii)].named_parameters()): #self.client_models[str(ii)].named_parameters()):

                if name != 'alphas':
                    if j == 0:
                        param.data = torch.zeros(param.shape).cuda()

                    param.data += alphas[ii]*param2.data # we add all participating client's models to the one here.

            j += 1

        self.client_models[str(i)] = zero_copy.double()

    def loop23(self,epochs):
        selected_clients = [5,6,7,8]  # [0,1, 2,3,4, 5, 6,7,8,9]
        #graph = self.create_graph(selected_clients)
        list1 = []
        self.alphas = torch.zeros(self.total_clients, self.total_clients)
        self.update_local_models(selected_clients) #### put this back in place later!!!!

        # Train local models
        for epoch in range(epochs):
            p = 0
            print("epoch: ",epoch)
            # train local model


            for i in selected_clients:
                print("client: ",i)

                # evaluate performance before combining models
                loss_test = self.evaluate(i,self.client_models[str(i)])
                print(loss_test)

                # initiate dataloader
                dataloader = self.dataloaders[str(i)]

                client_model_dict, optimizer_alpha = self.init_alpha(i,selected_clients)

                # store loss and alpha for plotting later on
                train_losses = []
                alpha = []

                other_clients = [iii for iii in selected_clients if iii != i]

                for _ in range(1):
                    losses = 0
                    for batch_idx, (data, target) in enumerate(dataloader):

                        data = data.double().cuda()
                        target = target.long().cuda()
                        optimizer_alpha.zero_grad()

                        output = self.client_models[str(i)](data,i, client_model_dict,client_numbers=other_clients)

                        # regularizer
                        uniform_alpha = (1/(len(selected_clients)))*torch.ones(len(selected_clients),1).cuda().double()
                        real_alphas = self.client_models[str(i)].alphas[selected_clients]
                        uniform_loss = F.mse_loss(uniform_alpha,real_alphas)

                        loss = F.nll_loss(output, target)  #+ uniform_loss

                        loss.backward()


                        optimizer_alpha.step()
                        losses += loss.detach().cpu().numpy()

                        self.client_models[str(i)].alphas.requires_grad = False
                        # print('before: ',local_param)
                        self.client_models[str(i)].alphas.data = torch.clamp(self.client_models[str(i)].alphas.data,0,1)
                        self.client_models[str(i)].alphas.requires_grad = True


                    #print("alpha", self.client_models[str(i)].alphas[5].detach().cpu().numpy())

                    train_losses.append(losses)
                    alpha.append(self.client_models[str(i)].alphas[5].detach().cpu().numpy())

                    #print("train loss: ",losses)

                # Use the optimal alpha to adjust the local model
                self.combine_models(i,other_clients)

                # Gather alphas so we can monitor them
                self.alphas[i,:] = self.client_models[str(i)].alphas.T
                self.alphas[i,i] = 1 - torch.sum(
                    torch.tensor([iii for idx, iii in enumerate(self.client_models[str(i)].alphas) if idx != i and idx in other_clients]))
            np.set_printoptions(precision=2)
            print(self.alphas[5:9,5:9].detach().cpu().numpy())
        plot_loss_curves(train_losses, alpha)



### OLD CODE:
    def pairwise_angles(self, selected_clients):
        angles = torch.zeros([self.total_clients, self.total_clients])
        print(angles.shape)
        for i in selected_clients:
            for j in selected_clients:
                list1 = []  # [copy.deepcopy(self.client_models[str(client)]).parameters() for client in selected_clients]
                list2 = []
                for value1, value2 in zip(self.client_models[str(i)].parameters(),
                                          self.client_models[str(j)].parameters()):
                    list1.append(value1.flatten())
                    list2.append(value2.flatten())
                s1 = torch.cat(list1)

                s2 = torch.cat(list2)
                angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-8)
        return angles.detach().numpy()

    def update_alpha_gd(self, ii, selected_clients, graph, alpha_init=torch.tensor([0.33, 0.33]).cuda()):
        alphas = torch.nn.Parameter(alpha_init, requires_grad=True)
        optimizer_alpha = torch.optim.Adam([alphas])
        local_graph = graph[ii]
        # print(local_graph)
        i = selected_clients[ii]
        dataloader = self.dataloaders_test[str(i)]
        local_model = self.client_models[str(i)]
        neigh1 = self.client_models[str(local_graph[0])]
        neigh2 = copy.deepcopy(self.client_models[str(local_graph[1])])

        loss_test = 0
        self.client_models[str(i)].eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target = target.long().cuda()

                output = local_model(data)
                loss_test += F.nll_loss(output, target)

        loss_test = 0
        loss_test2 = 0
        losses1 = 0
        losses2 = 0
        losses3 = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target = target.long().cuda()

                # output = local_model(data)
                output = local_model(data)
                loss_test2 += F.nll_loss(output, target)
                # output = alphas[0]*neigh1(data) + alphas[1]*neigh2(data) + (1-alphas[0]-alphas[1])*local_model(data)
                loss1 = F.nll_loss(neigh1(data), target)
                # loss2 = F.nll_loss(neigh2(data),target)
                loss3 = F.nll_loss(local_model(data), target)
                losses1 += loss1
                # losses2 += loss2
                losses3 += loss3

        return np.array([alphas, alphas])

    def update_alpha_cos(self, selected_clients,graph):
        # lphas = torch.nn.Parameter(alpha_init, requires_grad=True)
        # optimizer_alpha = torch.optim.Adam([alphas])
        # local_graph = graph[ii]
        # i = selected_clients[ii]
        # dataloader = self.dataloaders[str(i)]
        print(selected_clients)
        alphas = self.pairwise_angles(
            selected_clients)  # [copy.deepcopy(self.client_models[str(client)]).parameters() for client in selected_clients])
        print(alphas)
        return alphas



    def loop(self, epochs, mode='disjoint'):
        selected_clients = [1, 2, 5, 6]
        graph = self.create_graph(selected_clients)
        list1 = []
        self.alphas = torch.ones(self.total_clients, self.total_clients) * (1 / 3)

        for epoch in range(epochs):
            p = 0
            for i in selected_clients:
                self.update_local_models(selected_clients)
                local_graph = graph[p]
                dataloader = self.dataloaders_test[str(i)]
                local_model = self.client_models[str(i)]

                loss_test = 0
                local_model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(dataloader):
                        data = data.double().cuda()
                        target = target.long().cuda()

                        output = local_model(data)
                        loss_test += F.nll_loss(output, target)
                p += 1
                print(loss_test)

            alpha_init = torch.tensor([0, 0.1, 0.1]).double().cuda()
            alphas = torch.nn.Parameter(alpha_init, requires_grad=True)
            net1 = Net3('MNIST_niid', alphas).cuda()


            for local_param,local_param2,local_param3,local_param4 in zip(self.client_models[str(1)].parameters(),
                                                             self.client_models[str(2)].parameters(),
                                                             self.client_models[str(5)].parameters(),
                                                             self.client_models[str(6)].parameters()):
              #local_param.data = 0.25*local_param.data + 0.25*local_param2.data + 0.25*local_param3.data + 0.25*local_param4.data #*(1-alphas[0]-alphas[1]) + local_param2.data*alphas[0] + local_param3.data*alphas[1]
              local_param.data =  0 * local_param3.data + 1* local_param4.data  # *(1-alphas[0]-alphas[1]) + local_param2.data*alphas[0] + local_param3.data*alphas[1]
            ii = 0
            for (name1,local_param), (name2,local_param2) in zip(self.client_models[str(6)].named_parameters(),net1.named_parameters()):
              #local_param2.data = local_param.data
              if ii != 0:
                  local_param2.data = param_store
                  local_param2.requires_grad = False
              param_store = local_param.data
              print(name1)
              print(name2)
              ii = ii + 1

              print(ii)

            print(net1.fc2.bias)
            print(param_store)
            net1.fc2.bias.data = param_store
            net1.fc2.bias.requires_grad = False
            print(net1.fc2.bias)


            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = self.client_models[str(6)](data)
                    loss_test += F.nll_loss(output, target)
            print("before combining models")
            print(loss_test)

            #self.client_models[str(1)].eval()
            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = self.client_models[str(1)](data)
                    loss_test += F.nll_loss(output, target)
            print("after combining models")
            print(loss_test)


            print("alphas before gradient descent")
            print(alphas)
            net1.double()

            net1.eval()

            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = net1(data,self.client_models[str(5)])
                    loss_test += F.nll_loss(output, target)
            print("after combining models 2 ")
            print(loss_test)

            loss_test = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = self.client_models[str(6)](data)
                    loss_test += F.nll_loss(output, target)
            print("before combining models")
            print(loss_test)

            breakpoint()

    def loop3(self,epochs):
        selected_clients = [1, 2, 5, 6]
        graph = self.create_graph(selected_clients)
        list1 = []
        self.alphas = torch.ones(self.total_clients, self.total_clients) * (1 / 3)

        for epoch in range(epochs):
            p = 0
            for i in selected_clients:
                self.update_local_models(selected_clients)
                local_graph = graph[p]
                dataloader = self.dataloaders_test[str(i)]
                local_model = self.client_models[str(i)]

                loss_test = 0
                local_model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(dataloader):
                        data = data.double().cuda()
                        target = target.long().cuda()

                        output = local_model(data)
                        loss_test += F.nll_loss(output, target)
                p += 1
                print(loss_test)

        alpha_init = torch.tensor([0, 0.1, 0.1]).double().cuda()
        alphas = torch.nn.Parameter(alpha_init, requires_grad=True)
        net1 = Net4('MNIST_niid', alphas).cuda()

        print(net1.conv1.weight.requires_grad)

        ii = 0
        for (name1, local_param), (name2, local_param2) in zip(self.client_models[str(6)].named_parameters(),
                                                               net1.named_parameters()):
            if ii != 0:
                local_param2.data = param_store
                local_param2.requires_grad = False
            param_store = local_param.data
            ii = ii + 1

        print(self.client_models[str(6)].fc2.bias)
        #print(param_store)
        net1.fc2.bias.data = param_store
        net1.fc2.bias.requires_grad = False
        print(net1.fc2.bias)
        net1.eval()

        optimizer_alpha = torch.optim.Adam(net1.parameters(),lr=0.01)
        train_losses = []
        alpha = []
        for iii in range(30):
            losses = 0
            grads = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.double().cuda()
                target = target.long().cuda()

                optimizer_alpha.zero_grad()

                output = net1(data, self.client_models[str(5)].eval())  # self.client_models[str(1)](data)
                loss = F.nll_loss(output, target)

                loss.backward()
                grads += torch.sum(net1.alphas.grad)

                optimizer_alpha.step()

                losses += loss.detach().cpu().numpy()



            print("alpha", net1.alphas[0].detach().cpu().numpy())
            #print('epoch: ', iii)
            train_losses.append(losses)
            alpha.append(net1.alphas[0].detach().cpu().numpy())
            dataloader_test = self.dataloaders_test[str(1)]
            with torch.no_grad():
                loss_test = 0
                for batch_idx, (data, target) in enumerate(dataloader_test):
                    data = data.double().cuda()
                    target = target.long().cuda()

                    output = net1(data, self.client_models[str(5)].eval())
                    loss_test += F.nll_loss(output, target).detach().cpu().numpy()
            print("train loss: ",losses)
            print("test loss: ",loss_test)
        print(net1.alphas[0])
        plot_loss_curves(train_losses, alpha)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from tensorflow import keras
    from data import FEMNIST_dataset, Partition_MNIST_NIID, MNIST_NIID_dataset
    import copy
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import random
    torch.manual_seed(42)
    np.random.seed(0)
    print(torch.cuda.is_available)

    #net1 = Net('MNIST_niid')
    #net2 = Net('MNIST_niid')

    #for param1,param2 in zip(net1.parameters(),net2.parameters()):
   #     param1 = param1 + param2

    train_data, test_data = keras.datasets.mnist.load_data()
    instance = Partition_MNIST_NIID(train_data[0], train_data[1])
    classes_per_user = 10
    total_clients = 10
    train_partition = instance.create_partition(10, classes_per_user, total_clients)
    test_instance = Partition_MNIST_NIID(test_data[0], test_data[1])
    test_partition = test_instance.create_partition_test(instance.sample_array)
    p2p = P2P_AFPL(total_clients, train_data, train_partition, test_data, test_partition)
    alphas = p2p.loop23(10)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
