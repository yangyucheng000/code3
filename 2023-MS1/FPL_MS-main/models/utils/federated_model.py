import numpy as np

from argparse import Namespace
from utils.conf import checkpoint_path
from utils.util import create_if_not_exists
import os
import mindspore
from mindspore import nn, Tensor
from mindspore.dataset import transforms 



class FederatedModel(nn.Cell):
    """
    Federated learning model.
    """
    NAME = None
    N_CLASS = None

    def __init__(self, nets_list: list,
                 args: Namespace, transform: transforms) -> None:
        super(FederatedModel, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.transform = transform

        # For Online
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.args.parti_num * self.args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.trainloaders = None 
        self.testlodaers = None
        self.trainloaders_size = None
        self.epoch_index = 0 
        self.checkpoint_path = checkpoint_path() + self.args.dataset + '/' + self.args.structure + '/'
        create_if_not_exists(self.checkpoint_path)

    def construct(self, x: Tensor) -> Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass

    def col_update(self, communication_idx, publoader):
        pass

    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                param_dict = mindspore.load_checkpoint(save_path)
                mindspore.load_param_into_net(self.nets_list[j], param_dict)
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            net_para = net.parameters_dict()
            prev_net = prev_nets_list[net_id]
            mindspore.load_param_into_net(prev_net,net_para)

    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        online_clients = self.online_clients
        global_w = self.global_net.parameters_dict()
  
        if freq == None and self.args.averaging == 'weight':
            online_clients_len = []
            for i in range(len(online_clients)):
                #online_clients_len.append(self.trainloaders[i].get_dataset_size() * self.trainloaders[i].get_batch_size())
                online_clients_len.append(self.trainloaders_size[i])
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        elif freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        first = True
        for net_id in online_clients:
            net = self.nets_list[net_id]
            net_para = net.parameters_dict()

            if first:
                first = False
                for key in net_para:
                    global_key = 'global_net.' + key
                    update = nn.ParameterUpdate(global_w[global_key])
                    new_val = net_para[key] * freq[net_id]
                    update(new_val)

            else:
                for key in net_para:
                    global_key = 'global_net.' + key
                    update = nn.ParameterUpdate(global_w[global_key])
                    new_val = global_w[global_key] + net_para[key] * freq[net_id]
                    update(new_val)


        mindspore.load_param_into_net(global_net, global_w)
        
        for _, net in enumerate(self.nets_list):
            net_w = global_net.parameters_dict()
            mindspore.load_param_into_net(net, net_w)

