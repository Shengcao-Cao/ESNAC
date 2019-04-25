import torch
import datasets
import models

# global
device = 'cuda'                                             # used device, which can be 'cpu' or 'cuda'
model = './models/pretrained/resnet34_cifar100.pth'         # pretrained teacher model
savedir = './save/resnet34_cifar100_0'                      # save directory
writer = None                                               # record writer for tensorboardX
i = 0                                                       # sample index in search

# acquisition.py
ac_search_n = 1000                                          # number of randomly sampled architectures when optimizing acquisition function (see 3.2)

# architecture.py
ar_max_layers = 128                                         # maximum number of layers of the original architecture
ar_channel_mul = 2                                          # numbers of channels in layers should be divisible by this parameter
                                                            # necessary for ShuffleNet which has group conv, not used in VGG/ResNet
# hyper-params for random sampling in search space (see 3.2 & 6.5)
ar_p1 = [0.3, 0.4, 0.5, 0.6, 0.7]                           # for layer removal
ar_p2 = [0.0, 1.0]                                          # for layer shrinkage
ar_p3 = [0.003, 0.005, 0.01, 0.03, 0.05]                    # for adding skip connections

# compression.py
# hyper-params for multiple kernel strategy (see 3.3 & 6.3)
co_step_n = 20                                              # number of search steps
co_kernel_n = 8                                             # number of kernels, as well as evaluated architectures in each search step
co_best_n = 4                                               # number of saved best architectures during search, all of which will be fully trained
co_graph_gen = 'get_graph_resnet'                           # how to generate computation graph of original architecture
# hyper-params for stopping criterion of kernel optimization
co_alpha = 0.5
co_beta = 0.001
co_gamma = 0.5

# kernel.py
# hyper-params for kernels (see 3.1)
ke_alpha = 0.01
ke_beta = 0.05
ke_gamma = 1
ke_input_size = 16 + ar_max_layers * 2
ke_hidden_size = 64
ke_num_layers = 4
ke_bidirectional = True
ke_lr = 0.001
ke_weight_decay = 5e-4

# training.py
# hyper-params for training during *search*
tr_se_optimization = 'Adam'
tr_se_epochs = 10
tr_se_lr = 0.001
tr_se_momentum = 0.9
tr_se_weight_decay = 5e-4
tr_se_lr_schedule = None
tr_se_loss_criterion = 'KD'                                 # 'KD': knowledge distillation using teacher outputs
                                                            # 'CE': cross entropy using original labels
# hyper-params for *fully* training after search
tr_fu_optimization = 'SGD'
tr_fu_epochs = 300
tr_fu_lr = 0.01
tr_fu_momentum = 0.9
tr_fu_weight_decay = 5e-4
tr_fu_lr_schedule = 'step'
tr_fu_from_scratch = False                                  # False: continue training based on simple training results during search
                                                            # True: re-initialize weights and train from scratch