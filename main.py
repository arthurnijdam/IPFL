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



def training_loop(n_epochs,dataset,log_interval,n_clients,total_clients,local_iterations,save_dir,type,classes_per_user):
    if type == 'federated':
        model_object = Federated_Averaging('cuda:0', Net(dataset=dataset), n_clients, total_clients, dataset=dataset,classes_per_user=classes_per_user)
    if type == 'global':
        model_object = Global_Model('cuda:0', Net(dataset=dataset), n_clients, total_clients, dataset=dataset,classes_per_user=classes_per_user)
    if type == 'afpl':
        model_object = AFPL('cuda:0', Net(dataset=dataset), n_clients, total_clients, dataset=dataset,classes_per_user=classes_per_user)
    train_losses = []
    test_losses = []
    best_test_loss = 1000000
    for i in range(n_epochs):
        if type == 'federated' or type == 'afpl':
            model_object.setup(epoch=i)
            #print(model_object.selected_clients)
            #breakpoint()
            train_loss = model_object.train_one_epoch(epoch=i,local_iterations=local_iterations)
        else:
            train_loss = model_object.train_one_epoch()
        train_losses.append(train_loss)
        test_loss = model_object.evaluate_one_epoch()
        test_losses.append(model_object.accuracy)
        if test_loss < best_test_loss:
            torch.save(model_object.global_model.state_dict(), os.path.join(save_dir, 'model', 'best_model.pt'))

        print('train loss: ', train_loss)
        print('test loss: ', model_object.accuracy)
        # torch.save(federated_averaging.global_model.state_dict(),os.path.join(save_dir,'model',str(i)+'_model.pt'))
        with open(os.path.join(save_dir, 'loss.txt'), 'w') as f:
            for i, _ in enumerate(train_losses):
                if type != 'afpl':
                    f.write('epoch: ' + str(
                    i) + ' train loss: ' + f"{train_losses[i]}" + ' test loss: ' + f"{test_losses[i]}\n")
                else:
                    f.write('epoch: ' + str(
                        i) + ' train loss global: ' + f"{train_losses[i][0]}" + ' train loss local: ' + f"{train_losses[i][1]}"+' alpha: ' + f"{train_losses[i][2]}"+ ' test loss: ' + f"{test_losses[i]}\n")


def training_loop_local(n_epochs,dataset,log_interval,n_clients,total_clients,save_dir,classes_per_user):
    model_object = Local_Model('cuda:0', Net(dataset=dataset), n_clients, total_clients, dataset=dataset,classes_per_user=classes_per_user)
    best_client_models = model_object.train_loop(n_epochs=n_epochs,save_dir=save_dir)
    torch.save(best_client_models, os.path.join(save_dir, 'model', 'best_model.pt'))

def plot_loss_curves(experiment_name='federated_FEMNIST_test'):
    train_losses = []
    test_losses = []
    with open(os.path.join('checkpoints', experiment_name, 'loss.txt'), 'r') as file:
        for line in file:
            _, train, test = line.split("loss: ")
            train_losses.append(float(train.split(" ", 1)[0]))
            test_losses.append(float(test.split("\n", 1)[0]))
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join('checkpoints', experiment_name, 'loss_curve.png'))
    plt.show()
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
    if training_type == 'local':
        training_loop_local(n_epochs,dataset,log_interval,n_clients,total_clients,save_dir,classes_per_user)
    else:
        training_loop(n_epochs, dataset, log_interval, n_clients, total_clients, local_iterations, save_dir,training_type,classes_per_user)
    plot_loss_curves(settings['experiment_name'])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
