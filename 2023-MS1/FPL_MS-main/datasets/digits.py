from utils.conf import data_path
from datasets.utils.federated_dataset import FederatedDataset, partition_digits_domain_skew_loaders
from datasets.transforms.denormalization import DeNormalize
from backbone.ResNet import resnet10, resnet12, resnet152

import mindspore
import numpy as np
from mindspore import dataset, Tensor
from mindspore.dataset import MnistDataset, USPSDataset, SVHNDataset, ImageFolderDataset
from mindspore.dataset import vision
from mindspore.dataset import transforms


class DuplicateChannels:
    def __call__(self, img):
        img = np.array(img) 
        img = np.repeat([img], 3, axis=0)
        return img


ini_len_dict = {}
not_used_index_dict = {}

def get_sampler(dataset_name):
    
    percent_dict = {'mnist': 0.01, 'usps': 0.01, 'svhn': 0.01, 'syn': 0.01}
    dataset_size_dict = {'mnist': 60000, 'usps': 7291, 'svhn': 73257, 'syn': 10000}
    
    name = dataset_name
    dataset_size = dataset_size_dict[name]
    
    if name not in not_used_index_dict:
        not_used_index_dict[name] = np.arange(dataset_size)
        ini_len_dict[name] = dataset_size
    
    idxs = np.random.permutation(not_used_index_dict[name])

    percent = percent_dict[name]
    
    num_per_client = int(percent * ini_len_dict[name])

    
    selected_idx = idxs[0:num_per_client]

    not_used_index_dict[name] = idxs[num_per_client:]

    train_sampler = dataset.SubsetRandomSampler(selected_idx)
    
    return train_sampler


class MyDigits(dataset.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([vision.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = self.__build_truncated_dataset__()
        
        
    
    def __build_truncated_dataset__(self):
        
        sampler = get_sampler(self.data_name)
        if self.data_name == 'mnist':
            path = self.root + "MNIST/raw/"
            dataobj = MnistDataset(path, usage='train' if self.train else 'test', sampler=sampler if self.train else None) 
        elif self.data_name == 'usps':
            dataobj = USPSDataset(self.root, usage='train' if self.train else 'test', num_samples=73 if self.train else None)
        elif self.data_name == 'svhn':
            dataobj = SVHNDataset(self.root, usage='train' if self.train else 'test', sampler=sampler if self.train else None)
            
        dataobj = dataobj.map(operations=self.transform, input_columns='image')
        dataobj = dataobj.map(self.target_transform, 'label')
        return dataobj


class ImageFolder_Custom(dataset.Dataset):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            folder_path = self.root + self.data_name + '/train/'
        else:
            folder_path = self.root + self.data_name + '/val/'
        
        sampler = get_sampler(self.data_name)
        imagefolder_obj = ImageFolderDataset(folder_path, sampler=sampler if self.train else None, decode=True)
        imagefolder_obj = imagefolder_obj.map(operations=self.transform, input_columns='image')
        self.imagefolder_obj = imagefolder_obj.map(self.target_transform, 'label')

        
class FedLeaDigits(FederatedDataset):
    NAME = 'fl_digits'
    SETTING = 'domain_skew'
    DOMAINS_LIST = ['mnist', 'usps', 'svhn', 'syn']
    percent_dict = {'mnist': 0.01, 'usps': 0.01, 'svhn': 0.01, 'syn': 0.01}

    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    SYN_Nor_TRANSFORM = transforms.Compose(
        [vision.ToPIL(),
         vision.Resize((32, 32)),
         vision.ToTensor(),
         vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
         ])
    Nor_TRANSFORM = transforms.Compose(
        [vision.ToPIL(),
         vision.Resize((32, 32)),
         vision.ToTensor(),
         vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
         ])

    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [vision.ToPIL(),
         vision.Resize((32, 32)),
         vision.Rescale(1.0 / 255.0, 0),
         DuplicateChannels(),
         vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
        ])

    def get_data_loaders(self, selected_domain_list=[]):

        using_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list

        nor_transform = self.Nor_TRANSFORM
        sin_chan_nor_transform = self.Singel_Channel_Nor_TRANSFORM

        train_dataset_list = []
        test_dataset_list = []
        dataloader_size = []
        
        label_transform = transforms.TypeCast(mindspore.int32)
        test_label_transform = transforms.TypeCast(mindspore.int32)
        

        for _, domain in enumerate(using_list):
            if domain == 'syn':
                train_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=True,
                                                   transform=self.SYN_Nor_TRANSFORM,target_transform=label_transform)
                dataloader_size.append(100)
            else:
                if domain in ['mnist', 'usps']:
                    train_dataset = MyDigits(data_path(), train=True,
                                             transform=sin_chan_nor_transform, target_transform=label_transform, data_name=domain)
                    dataloader_size.append(600 if domain == 'mnist' else 73)
                else:
                    train_dataset = MyDigits(data_path(), train=True,
                                             transform=nor_transform, target_transform=label_transform, data_name=domain)
                    dataloader_size.append(733)
                    
            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            if domain == 'syn':
                test_dataset = ImageFolder_Custom(data_name=domain, root=data_path(), train=False,
                                                  transform=self.SYN_Nor_TRANSFORM,target_transform=test_label_transform)
            else:
                if domain in ['mnist', 'usps']:
                    test_dataset = MyDigits(data_path(), train=False,
                                            transform=sin_chan_nor_transform, target_transform=test_label_transform, data_name=domain)
                else:

                    test_dataset = MyDigits(data_path(), train=False,
                                            transform=nor_transform, target_transform=test_label_transform, data_name=domain)

            test_dataset_list.append(test_dataset)
            
        traindls, testdls = partition_digits_domain_skew_loaders(train_dataset_list, test_dataset_list, self)

        return traindls, testdls, dataloader_size

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [vision.ToPIL(), FedLeaDigits.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list):

        nets_dict = {'resnet10': resnet10, 'resnet12': resnet12}
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                nets_list.append(resnet10(FedLeaDigits.N_CLASS))
                
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaDigits.N_CLASS))

        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = vision.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
