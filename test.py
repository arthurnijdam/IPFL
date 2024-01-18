# This file is to test

import yaml
import shutil
from models import Federated_Averaging,Net,Global_Model,Local_Model,AFPL
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def init():
    with open('settings/train_settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir(os.path.join('checkpoints', settings['experiment_name'])):
        os.mkdir(os.path.join('checkpoints', settings['experiment_name']))
    save_dir = os.path.join('checkpoints', settings['experiment_name'])
    if not os.path.isdir(os.path.join(save_dir, 'model')):
        os.mkdir(os.path.join(save_dir, 'model'))
    shutil.copyfile('settings/train_settings.yaml', save_dir + '/train_settings.yaml')
    return settings,save_dir

def different_degrees_of_IID_global():
    experiment_name = "global_MNIST_IID"
    with open(os.path.join('checkpoints', experiment_name, 'train_settings.yaml'), 'r') as file:
        settings = yaml.safe_load(file)
        n_clients = settings['total_clients']  # settings['n_clients']
        total_clients = settings['total_clients']
        dataset = settings['Dataset']
        classes_per_user = settings['classes_per_user']
    model_path = os.path.join('checkpoints', experiment_name, 'model', 'best_model.pt')
    model = Net(dataset=dataset)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_object = Global_Model('cuda:0', model, n_clients, total_clients, dataset=dataset,
                                       classes_per_user=classes_per_user)

    targets, outputs = model_object.evaluate_one_epoch(calc_accuracy=True)
    confusion_matrix = calc_confusion_matrix(targets,outputs,print_result=False)
    print('accuracy global model: ')
    accuracy(confusion_matrix)


def calc_confusion_matrix(targets,outputs,print_result=True):
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

def accuracy(confusion_matrix,print_result=True,return_t=False):
    correct = np.trace(confusion_matrix)
    total = np.sum(confusion_matrix)
    if print_result == True:
        #print(correct)
        #print(total)
        print('accuracy = ',correct/total*100,'%')
    if return_t == True:
        return correct/total*100

def plot_loss_curves(experiment_name='federated_FEMNIST_test'):
    train_losses = []
    test_losses = []
    with open(os.path.join('checkpoints', experiment_name, 'loss.txt'), 'r') as file:
        for line in file:
            _,train,test = line.split("loss: ")
            train_losses.append(float(train.split(" ",1)[0]))
            test_losses.append(float(test.split("\n",1)[0]))
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    plt.figure()
    plt.plot(train_losses,label="train loss")
    plt.plot(test_losses,label="test loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join('checkpoints',experiment_name,'loss_curve.png'))
    plt.show()






def different_degrees_of_IID(type='federated'):
    for i in range(1,11):
        experiment_name = type +"_MNIST_IID_" + str(i)
        with open(os.path.join('checkpoints', experiment_name, 'train_settings.yaml'), 'r') as file:
            settings = yaml.safe_load(file)
            n_clients = settings['total_clients'] #settings['n_clients']
            total_clients = settings['total_clients']
            dataset = settings['Dataset']
            classes_per_user = settings['classes_per_user']
        model_path = os.path.join('checkpoints',experiment_name,'model','best_model.pt')
        model = Net(dataset=dataset)
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        if type == 'federated':
            model_object = Federated_Averaging('cuda:0', model, n_clients, total_clients, dataset=dataset,
                                           classes_per_user=classes_per_user)
        if type == 'afpl':
            model_object = AFPL('cuda:0', model, n_clients, total_clients, dataset=dataset,
                                               classes_per_user=classes_per_user)
        model_object.setup()
        targets, outputs = model_object.evaluate_one_epoch(calc_accuracy=True)
        confusion_matrix = calc_confusion_matrix(targets,outputs,print_result=False)
        print('accuracy with ',str(i),' classes per user:')
        accuracy(confusion_matrix)

def different_degrees_of_IID_local():
    for i in range(0,1):

        experiment_name = "test" #"local_MNIST_IID_test"# + str(i)
        with open(os.path.join('checkpoints', experiment_name, 'train_settings.yaml'), 'r') as file:
            settings = yaml.safe_load(file)
            n_clients = settings['total_clients']  # settings['n_clients']
            total_clients = settings['total_clients']
            dataset = settings['Dataset']
            classes_per_user = settings['classes_per_user']
        model_path = os.path.join('checkpoints', experiment_name, 'model', 'best_model.pt')
        best_local_models = torch.load(model_path, map_location=torch.device('cpu'))
        total_accuracy = np.zeros((n_clients, 1))
        for key in best_local_models.keys():
            model = Net(dataset=dataset)
            model.load_state_dict(best_local_models[key])

            model_object = Local_Model('cuda:0', model, n_clients, total_clients, dataset=dataset,
                                           classes_per_user=classes_per_user)
            number = int(key[7:])
            model = model.double().to('cuda:0')
            targets,outputs = model_object.evaluation_loop(model,number)
            confusion_matrix = calc_confusion_matrix(targets, outputs, print_result=False)
            #print('accuracy for local model: ', key)
            accuracy_sample = accuracy(confusion_matrix,print_result=True,return_t=True)
            total_accuracy[number]=accuracy_sample
            #breakpoint()
        print('accuracy: ',np.mean(total_accuracy))






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(0)
    print(torch.cuda.is_available)
    settings,save_dir = init()
    n_epochs = settings['n_epochs']
    log_interval = settings['log_interval']
    n_clients = settings['n_clients']
    total_clients = settings['total_clients']
    dataset = settings['Dataset']
    local_iterations = settings['local_iterations']
    training_type = settings['type']
    classes_per_user = settings['classes_per_user']
    #plot_loss_curves()
    #different_degrees_of_IID(type='afpl')
    #different_degrees_of_IID_global()
    different_degrees_of_IID_local()

    #(dataset, n_clients, total_clients, save_dir, classes_per_user)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/