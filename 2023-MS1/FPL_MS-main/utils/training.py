from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
import numpy as np
from utils.logger import CsvWriter
from collections import Counter
import mindspore
from mindspore.train import Model
import copy
    
def global_evaluate(model: FederatedModel, test_dl: list, setting: str, name: str) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    net.set_train(mode=False)
    for j, dl in enumerate(test_dl):
        eval_net = Model(net, loss_fn=mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean'), metrics={'acc'})
        acc = eval_net.eval(dl)

        accs.append(round(100 * acc['acc'], 2))

    net.set_train(mode=True)
    return accs


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = {'mnist': 6, 'usps': 4, 'svhn': 3, 'syn': 7}  # base
        # selected_domain_dict = {'mnist': 0, 'usps': 0, 'svhn': 0, 'syn': 4}  # base
        # selected_domain_dict = {'mnist': 1, 'usps': 1, 'svhn': 1, 'syn': 1}
        # selected_domain_dict = {'mnist': 1, 'usps': 1, 'svhn': 9, 'syn': 9}  # 20

        # selected_domain_dict = {'mnist': 3, 'usps': 2, 'svhn': 1, 'syn': 4}  # 10

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders, trainloaders_size = private_dataset.get_data_loaders(selected_domain_list)
        
    model.trainloaders = pri_train_loaders
    model.trainloaders_size = trainloaders_size
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            model.loc_update(pri_train_loaders)

        
        test_dl_temp = []
        for i in range(len(test_loaders)):
            test_dl_temp.append(copy.deepcopy(test_loaders[i]))
        accs = global_evaluate(model, test_dl_temp, private_dataset.SETTING, private_dataset.NAME)
        del test_dl_temp
        
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        print('The ' + str(epoch_index) + ' Communcation Accuracy:', str(mean_acc), 'Method:', model.args.model)
        print(accs)
        
        #mindspore.ms_memory_recycle()

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)
