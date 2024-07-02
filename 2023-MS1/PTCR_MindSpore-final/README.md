# PTCR_MindSpore

This is a MindSpore **Ascend** implementation of the PTCR proposed in *[Pyramidal Transformer with Conv-Patchify for Person Re-identification.](https://dl.acm.org/doi/10.1145/3503161.3548770)*

##  Environment

- Python 3.8
- MindSpore 2.0.0 (Ascend)
- mindcv

## Usage

- Organize datasets as below

```
├──"DATASETS.ROOT_DIR" in /src/PTCR_MindSpore.yaml
   ├──market1501
      ├──Market-1501
         ├──bounding_box_train
         ├──query
         ├──bounding_box_test
   ├──dukemtmc-reid
      ├──DukeMTMC-reID
         ├──bounding_box_train
         ├──query
         ├──bounding_box_test
   ├──cuhk03
      ├──cuhk03_release
         ├──cuhk-03.mat
         ├──cuhk03_new_protocol_config_labeled.mat
         ├──cuhk03_new_protocol_config_detected.mat
   ├──msmt17
      ├──MSMT17_V1
         ├──train
         ├──test
         ├──list_val.txt
         ├──list_train.txt
         ├──list_query.txt
         ├──list_query.txt
```

- Set your own "OUTPUT_DIR " in /src/PTCR_MindSpore.yaml
- Set your own "DATASETS.NAMES " in /src/PTCR_MindSpore.yaml：{market1501、dukemtmcreid、cuhk03、msmt17}
- Train

```
python train.py
```

- Test

```
python test.py
```

- Checkpoint
- you can download the pretrained weight for train from [Google Drive](https://drive.google.com/file/d/1xOE66uiuE_mMgV1oO4DdkSL2gjhjVDFk/view?usp=sharing)
- you can download the finetuned weight for evaluation from [Google Drive](https://drive.google.com/file/d/1QR__rOaaeAtL62xL1EogziF2xEGPG1W9/view?usp=sharing)
