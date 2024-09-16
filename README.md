### Dataset 
We use three different datasets: 
1. MNIST, partitioned Non-IID. The MNIST dataset can be accessed via torchvision (https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
   and partitioned Non-IID using the Partition_MNIST_NIID class in data/FEMNIST_datasets.py.
2. MIT-BIH. We use both the MIT-BIH Arrythmia (https://physionet.org/content/mitdb/1.0.0/) and the MIT-BIH Supraventricular Arrhythmia dataset (https://physionet.org/content/svdb/1.0.0/).
   Both datasets were extracted and pre-processed using the method described in https://github.com/chenmingTHU/ECG_UDA/tree/master, but without domain adaptation.
3. CHB-MIT. The dataset can be accessed here: https://physionet.org/content/chbmit/1.0.0/.

### Training 
All MNIST experiments were run via train_bandits.py and IFCA_MNIST.py (only IFCA). 
All MIT-BIH experiments were run via train_bandits_MIT_BIH_3class.py and IFCA_MIT.py (only IFCA). 
All CHB-MIT experiments were run via train_bandits_CHB_MIT.py and IFCA_CHB.py (only IFCA).
You will need to manually adjust the file train_bandits_MNIST.sh so that it runs the correct python file for your desired dataset. 

The specific settings for your experiment can be defined in settings/train_settings_bandits.yaml. 
Here, 
* experiment_name = the name of the folder in which all checkpoints and results will be saved to. 
* dataset         = the name of the dataset you're using (either MNIST_NIID, MIT_BIH or CHB_MIT). 
* n_clients       = the number of Federated Learning clients (dataset-dependent, 10 for MNIST_NIID, 122 for MIT_BIH and 2 for CHB_MIT). 
* n_clients_UCB   = zeta, the maximum number of clients one client can collaborate with. This is at most n_clients -1. 
* seed            = the random seed, set for reproducibility. 
* n_classes_total = the total number of classes for MNIST-NIID experiments. To reproduce the experiments in our paper, set this to 4. 
* n_classes_per_user = the number of classes per user for MNIST-NIID experiments. To reproduce the experiments in our paper, set this to 2. 
* data_fraction   = the fraction of data used for training. This was set to [0.1, 0.25, 0.5 and 1] for our MIT-BIH and CHB-MIT experiments. 
* alpha           = the alpha value used for APFL. This was set to 0.75 as recommended in the APFL paper. 
* k               = the number of clusters for IFCA. This was set to [5, 6 (MNIST), 10, 15 and 20] for our experiments. 
* network         = the network used for training. This variable is unused in the current version of the code. 
* n_epochs        = the number of epochs for training, default = 100. 
* local_iterations = the number of local iterations. 
* type            = The baseline method up for evaluation. Set to "centralized", "federated", "local", "prtfl", "afpl" (corresponds to APFL), or "bandits" (ours).


### Evaluation 
When you start training, a folder will automatically be saved with the following contents: 
* A copy of the settings you specified in train_settings_bandits.yaml 
* Model checkpoints per FL clients are saved to a model subfolder
* An image of the accuracy progression over time called accuracy_progression.png
* An image of the loss curve called loss_curve.png
* The raw test loss, saved to losses_test.txt. 
* the best accuracy in a .txt file called test_accuracy.txt and the client-specific accuracies will be saved to test_accuracies.txt.  
