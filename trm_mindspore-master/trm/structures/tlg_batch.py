from dataclasses import dataclass
import numpy as np
# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: np.array
    queries: list
    wordlens: list
    all_iou2d: list
    moments: list
    num_sentence: list
    sentences: list
    durations: list
    phrase: list
    