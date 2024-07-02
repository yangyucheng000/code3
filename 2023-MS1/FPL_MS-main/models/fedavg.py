from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

import mindspore 
from mindspore import nn

class FedAvG(FederatedModel):
    NAME = 'fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedAvG, self).__init__(nets_list,args,transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].parameters_dict()
        for _,net in enumerate(self.nets_list):
            mindspore.load_param_into_net(net, global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        net_temp = {}
        dl_temp = {}    
        for i in online_clients:
            net_temp[i] = copy.deepcopy(self.nets_list[i])
            dl_temp[i] = copy.deepcopy(priloader_list[i])
            
        for i in online_clients:
            self._train_net(i, net_temp[i], dl_temp[i])
            self.nets_list[i] = copy.deepcopy(net_temp[i])
            mindspore.ms_memory_recycle()
        
        self.aggregate_nets(None)
        return  None

    def _train_net(self,index,net,train_loader):
        optimizer = nn.SGD(net.trainable_params(), learning_rate=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        train_net = nn.TrainOneStepCell(nn.WithLossCell(net,criterion), optimizer=optimizer)
        train_net.set_train(True)
        
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for di in train_loader.create_dict_iterator():
                result = train_net(di["image"], di["label"])
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, result)
                
        

            
                

