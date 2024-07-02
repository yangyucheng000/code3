from abc import abstractmethod
from argparse import Namespace
from typing import Tuple


class FederatedDataset:
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders = []
        self.test_loader = []
        self.args = args

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]):
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list):
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform():
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform():
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform():
        pass


    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass



def partition_digits_domain_skew_loaders(train_datasets: list, test_datasets: list,
                                         setting: FederatedDataset) -> Tuple[list, list]:
    
    for index in range(len(train_datasets)):
        name = train_datasets[index].data_name

        if name == 'syn':
            train_dataset = train_datasets[index].imagefolder_obj
        else:
            train_dataset = train_datasets[index].dataset

        train_loader = train_dataset

        train_loader = train_loader.batch(batch_size=setting.args.local_batch_size)
        
        setting.train_loaders.append(train_loader)

    for index in range(len(test_datasets)):
        name = test_datasets[index].data_name
        if name == 'syn':
            test_dataset = test_datasets[index].imagefolder_obj
        else:
            test_dataset = test_datasets[index].dataset

        test_loader = test_dataset
        test_loader = test_loader.batch(batch_size=setting.args.local_batch_size)

        setting.test_loader.append(test_loader)

    return setting.train_loaders, setting.test_loader


