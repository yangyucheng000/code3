from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
from utils.finch import FINCH
import numpy as np

import mindspore
from mindspore import nn, Tensor, ops

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FedHierarchy.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

def agg_func(protos:dict):
    """
    Returns the average of the weights.
    """
    protos_avg = dict()
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto_sum = proto_list[0]
            for i in range(1, len(proto_list)):
                proto_sum += proto_list[i]
            protos_avg[label] = proto_sum / len(proto_list)
        else:
            protos_avg[label] = proto_list[0]

    return protos_avg

class FPL(FederatedModel):
    NAME = 'fpl'
    COMPATIBILITY = ['homogeneity']
    
    def __init__(self, nets_list, args, transform):
        super(FPL, self).__init__(nets_list, args, transform)
        self.global_protos = []
        self.local_protos = {}
        self.infoNCET = args.infoNCET
        self.loss2 = 0

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].parameters_dict()
        for _, net in enumerate(self.nets_list):
            mindspore.load_param_into_net(net, global_w)
            
            
    def proto_aggregation(self, local_protos_list):
        # local_protos_list contains local prototype of each client, 
        # and each local prototype contains average feature vector of each label
        agg_protos_label = dict()
        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(proto)
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def get_indices(self, label, all_global_protos_keys):
        pos_indices = 0
        neg_indices = []
        for i, k in enumerate(all_global_protos_keys):
            if k == label.item():
                pos_indices = i
            else:
                neg_indices.append(i)
                
        return pos_indices, neg_indices
    
    def hierarchical_info_loss(self, f_now, label, mean_f, all_global_protos_keys):
        

        pos_indices = 0
 
        for i, k in enumerate(all_global_protos_keys):
            if k == label.item():
                pos_indices = i

                

        mean_f_pos = Tensor(mean_f[pos_indices])
        f_now = Tensor(f_now)

        cu_info_loss = self.loss_mse(f_now, mean_f_pos)

        return cu_info_loss


    def calculate_infonce(self, f_now, label, all_f, all_global_protos_keys):
        pos_indices = 0
        neg_indices = []
        for i, k in enumerate(all_global_protos_keys):
            if k == label.item():
                pos_indices = i
            else:
                neg_indices.append(i)

        f_pos = Tensor(all_f[pos_indices][0]).reshape(1,512)
        f_neg = ops.cat([Tensor(all_f[i]).reshape(-1, 512) for i in neg_indices], axis=0)
        #aaa
        f_proto = ops.cat((f_pos, f_neg), axis=0)
        f_now = f_now.reshape(1,512)
        
        f_now_np = f_now.asnumpy()
        f_proto_np = f_proto.asnumpy()
        def cosine_similarity_numpy(vec_a, vec_b):
            dot_product = np.dot(vec_a, vec_b.T)
            norm_a = np.linalg.norm(vec_a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(vec_b, axis=1)
            return dot_product / (norm_a * norm_b)
        l_np = cosine_similarity_numpy(f_now_np, f_proto_np)
        l = Tensor(l_np)

        #l = ops.cosine_similarity(f_now, f_proto, dim=1)
        l = ops.div(l, self.infoNCET)
        
        exp_l = ops.exp(l).reshape(1, -1)

        pos_num = f_pos.shape[0]
        neg_num = f_neg.shape[0]
 
        pos_mask = Tensor([1] * pos_num + [0] * neg_num).reshape(1, -1)
        
        pos_l = exp_l * pos_mask
        sum_pos_l = ops.sum(pos_l, dim=1)
        sum_exp_l = ops.sum(exp_l, dim=1)
        infonce_loss = -ops.log(sum_pos_l / sum_exp_l)
        return Tensor(infonce_loss)


    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        
        net_temp = {}
        dl_temp = {}    
        for i in online_clients:
            net_temp[i] = copy.deepcopy(self.nets_list[i])
            dl_temp[i] = copy.deepcopy(priloader_list[i])
            
        for i in online_clients:
            # self._train_net(i, self.nets_list[i], priloader_list[i])
            self._train_net(i, net_temp[i], dl_temp[i])
            self.nets_list[i] = copy.deepcopy(net_temp[i])
            mindspore.ms_memory_recycle()
        
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader):

        if len(self.global_protos) != 0:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = self.global_protos[protos_key]
                all_f.append(copy.deepcopy(temp_f))
                mean_f.append(copy.deepcopy(np.mean(temp_f, axis=0)))
            all_f = [item.copy() for item in all_f]
            mean_f = [item.copy() for item in mean_f]
        else:
            all_f = []
            mean_f = []
            all_global_protos_keys = []
        

        optimizer = nn.SGD(net.trainable_params(), learning_rate=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion1 = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        criterion = CustomLoss(criterion1, self.loss2)
        self.loss_mse = mindspore.nn.MSELoss()
        train_net= nn.TrainOneStepCell(nn.WithLossCell(net,criterion), optimizer=optimizer)
        train_net.set_train(True)
        
        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:

            agg_protos_label = {}
            for di in train_loader.create_dict_iterator():
                images = di["image"]
                labels = di["label"]
                
                #   train_net.set_train(False)
                f = net.features(images)
                #train_net.set_train(True)

                
                if len(self.global_protos) == 0:
                    loss_InfoNCE = 0 
                else:
                    i = 0
                    loss_InfoNCE = None

                    for label in labels:
                        if label in all_global_protos_keys:
                            
                            f_now = f[i]
                            cu_info_loss = self.hierarchical_info_loss(f_now, label, mean_f, all_global_protos_keys)
                            xi_info_loss = self.calculate_infonce(f_now, label, all_f, all_global_protos_keys)
                            loss_instance = xi_info_loss + cu_info_loss
                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                
                self.loss2 = loss_InfoNCE

                result = train_net(images, labels)
                iterator.desc =  "Local Pariticipant %d Loss = %0.3f" % (index, result)
                

                if iter == self.local_epoch - 1:
                    labels_numpy = labels.asnumpy()
                    f_numpy = f.asnumpy()
                    for i in range(len(labels_numpy)):
                        if labels_numpy[i] in agg_protos_label:
                            agg_protos_label[labels_numpy[i]].append(f_numpy[i, :])
                        else:
                            agg_protos_label[labels_numpy[i]] = [f_numpy[i, :]]
        
        agg_protos = agg_func(agg_protos_label)
        self.local_protos[index] = agg_protos
        
        
class CustomLoss(nn.Cell):
    def __init__(self, criterion1, loss2):
        super(CustomLoss, self).__init__(auto_prefix=False)
        self.criterion1 = criterion1
        self.loss2 = loss2

    def construct(self, outputs, labels):
        loss1 = self.criterion1(outputs, labels)
        total_loss = loss1 + self.loss2
        return total_loss
    
