import os
import functools
from mindspore.communication import get_rank


distributed = False

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    if distributed:
        return get_rank()
    else:
        global_rank_id = os.getenv('RANK_ID', '0')
        return int(global_rank_id)


def execute_distributed():
    global distributed
    distributed = True