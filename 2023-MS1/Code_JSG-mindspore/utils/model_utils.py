# todo: 把使用torch的地方改为使用mindspore


import mindspore as ms
import mindspore.ops as ops
import numpy as np


def count_parameters(model, verbose=True):
    n_all = sum(p.numel() for p in model.get_parameters())
    n_trainable = sum(p.numel() for p in model.trainable_params())
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable

##############################################################
# added
def get_center_based_props(num_gauss_center, num_gauss_width, width_lower_bound=0.05, width_upper_bound=1.0):
    # lb = 1 / num_gauss_center / 2
    # gauss_center = torch.linspace(lb, 1 - lb , steps=num_gauss_center) # new
    gauss_center = ops.linspace(width_lower_bound/2, 1 - width_lower_bound/2, steps=num_gauss_center)
    # gauss_center = torch.linspace(0, 1 , steps=num_gauss_center)
    gauss_center = gauss_center.unsqueeze(-1).broadcast_to((-1, num_gauss_width)).reshape(-1)
    gauss_width = ops.linspace(width_lower_bound, width_upper_bound, steps=num_gauss_width).unsqueeze(0).broadcast_to((num_gauss_center, -1)).reshape(-1)

    return gauss_center, gauss_width

def get_sliding_window_based_props(map_size, step=1, width_lower_bound=0.05, width_upper_bound=1.0):
    '''
    input:
        map_size: int, the assumed length of the sequence
    '''
    centers = []
    widths = []
    # off_set = 1 / map_size / 2
    for i in range(1, map_size + 1, step): # 从2开始呢？
        count = map_size - i + 1
        lower_bound = max(i / map_size / 2 , 0) # - off_set
        upper_bound = 1 - lower_bound # + 2*off_set
        temp = np.linspace(lower_bound, upper_bound, count, endpoint=True).tolist()
        centers.extend(temp)
        
        width = i / map_size
        width = min(width_upper_bound, max(width, width_lower_bound)) # clamp width
        widths.extend([width] * count)


    gauss_center = ops.clamp(ms.Tensor(centers), min=0, max=1).astype(ms.float32)
    gauss_width = ms.Tensor(widths).astype(ms.float32)

    return gauss_center, gauss_width


# with traunc optional
def generate_gauss_weight(props_len, center, width, sigma=9, truncate=False):
    weight = ops.linspace(0, 1, props_len) # ori version, not consider the bias of position
    # lb = 1 / props_len / 2 # lower bound(relative position of the first clip's center)
    # weight = torch.linspace(lb, 1-lb, props_len) # new version TODO: test

    # weight = weight.view(1, -1).broadcast_to((center.shape[0], -1)) # ori, core dump
    weight = weight.unsqueeze(0).broadcast_to((center.shape[0], -1)) # 测试
    center = center.unsqueeze(-1)
    width = width.unsqueeze(-1).clamp(1e-2) / sigma

    w = 0.3989422804014327
    weight = w/width*ops.exp(-(weight-center)**2/(2*width**2))

    if truncate:
        # truncate
        left = ((center - width / 2) * props_len).clamp(min=0)
        left = left.astype(ms.int32)
        right = ((center + width / 2) * props_len).clamp(max=props_len-1)
        right = right.astype(ms.int32)
        for idx, w in enumerate(weight):
            weight[idx][:left[idx]] = 0
            weight[idx][right[idx]+1:] = 0

    return weight/weight.max(axis=-1, keepdims=True)

def generate_mean_weight(props_len, center, width):
    weight = ops.ones((center.shape[0], props_len)).astype(center.dtype)
    left = ((center - width / 2) * props_len).clamp(min=0)
    left = left.astype(ms.int64)
    right = ((center + width / 2) * props_len).clamp(max=props_len-1)
    right = right.astype(ms.int64)
    for idx, w in enumerate(weight):
        weight[idx][:left[idx]] = 0
        weight[idx][right[idx]+1:] = 0
    
    return weight/weight.max(axis=-1, keepdims=True)

def get_center_from_center_indices(center_indices, num_gauss_center, width_lower_bound=0.05):
    centers = ops.linspace(width_lower_bound/2, 1 - width_lower_bound/2, steps=num_gauss_center).astype(ms.float32)
    shape = center_indices.shape
    if len(shape) == 1:
        centers = centers.unsqueeze(0).broadcast_to((shape[0], -1))
    elif len(shape) == 2:
        centers = centers.unsqueeze(0).unsqueeze(0).broadcast_to((shape[0], shape[1], -1))
    elif len(shape) == 3:
        centers = centers.unsqueeze(0).unsqueeze(0).unsqueeze(0).broadcast_to((shape[0], shape[1], shape[2], -1))
    elif len(shape) == 4:
        centers = centers.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).broadcast_to((shape[0], shape[1], shape[2], shape[3], -1))
    else:
        raise NotImplementedError("Only support 1, 2, 3, 4 dimensions")
    
    return centers.gather_elements(-1, center_indices.unsqueeze(-1)).squeeze(-1)

def get_width_from_width_indices(width_indices, num_width_center, width_lower_bound=0.05, width_upper_bound=1):
    widthes = ops.linspace(width_lower_bound, width_upper_bound, steps=num_width_center).astype(ms.float32)
    shape = width_indices.shape
    if len(shape) == 1:
        widthes = widthes.unsqueeze(0).broadcast_to((shape[0], -1))
    elif len(shape) == 2:
        widthes = widthes.unsqueeze(0).unsqueeze(0).broadcast_to((shape[0], shape[1], -1))
    elif len(shape) == 3:
        widthes = widthes.unsqueeze(0).unsqueeze(0).unsqueeze(0).broadcast_to((shape[0], shape[1], shape[2], -1))
    elif len(shape) == 4:
        widthes = widthes.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).broadcast_to((shape[0], shape[1], shape[2], shape[3], -1))
    else:
        raise NotImplementedError("Only support 1, 2, 3, 4 dimensions")
    
    return widthes.gather_elements(-1, width_indices.unsqueeze(-1)).squeeze(-1)

def get_gauss_props_from_clip_indices(indices, num_gauss_center, num_gauss_width, width_lower_bound=0.05, width_upper_bound=1):
    row, col = indices.shape
    centers = ops.linspace(width_lower_bound/2, 1 - width_lower_bound/2, steps=num_gauss_center).unsqueeze(0).unsqueeze(0).broadcast_to((row, col, num_gauss_center)).astype(ms.float32)
    widthes = ops.linspace(width_lower_bound, width_upper_bound, steps=num_gauss_width).unsqueeze(0).unsqueeze(0).broadcast_to((row, col, num_gauss_width)).astype(ms.float32)

    center_indices = indices // num_gauss_width
    width_indices = indices % num_gauss_width
    gauss_center = ops.gather_elements(centers, dim=-1, index=center_indices.unsqueeze(-1)).squeeze(-1)
    gauss_width = ops.gather_elements(widthes, dim=-1, index=width_indices.unsqueeze(-1)).squeeze(-1)

    return gauss_center, gauss_width

# new version
def get_props_from_indices(indices, gauss_center, gauss_width):
    '''
    input:
        indices: [row, col]
        gauss_center: [num_gauss_center]
        gauss_width: [num_gauss_width]
    output:
        center_prop: [row, col]
        width_prop: [row, col]
    '''
    row, col = indices.shape

    # gauss_center = gauss_center.to(indices.device)
    # gauss_width = gauss_width.to(indices.device)

    expanded_centers = gauss_center.unsqueeze(0).unsqueeze(0).broadcast_to((row, col, -1))
    expanded_widths = gauss_width.unsqueeze(0).unsqueeze(0).broadcast_to((row, col, -1))

    center_prop = ops.gather_elements(expanded_centers, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
    width_prop = ops.gather_elements(expanded_widths, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)

    return center_prop, width_prop

