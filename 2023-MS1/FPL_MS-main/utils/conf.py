import random
import numpy as np

import mindspore



def data_path() -> str:
    return './data/'


def base_path() -> str:
    return './data/'

def checkpoint_path() -> str:
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
