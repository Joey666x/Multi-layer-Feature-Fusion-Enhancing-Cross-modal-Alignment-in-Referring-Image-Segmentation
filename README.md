

This is an official PyTorch implementation .



# Preparation

## Environment
```bash
conda create -n MMFRIS python=3.9.18 -y
conda activate MMFRIS
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirement.txt
```

## Datasets
The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

## Pretrained weights
Download the pretrained weights of DiNOv2-B and ViT-B to pretrain
```bash
mkdir pretrain && cd pretrain
## DiNOv2-B
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth

## ViT-B
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

# Quick Start

To train MMFRIS, modify the script according to your requirements and run it:

```
bash run_scripts/train.sh
```

If you want to use multi-gpu training, simply modify the `gpu` in the run_scripts/train.sh. Please notice that you should execute this bash script under the first-level directory (the path with train.py).

To evaluate MMFRIS, specify the model file path in test.sh according to your requirements and run the script:

```
bash run_scripts/test.sh
```

If you want to visualize the results, simply modify the `visualize` to `True` in the config file. 



# Acknowledgements

The code is based on [CRIS](https://github.com/DerrickWang005/CRIS.pytorch), [ETRIS](https://github.com/kkakkkka/ETRIS) and [DiNOv2](https://github.com/facebookresearch/dinov2). We thank the authors for their open-sourced code and encourage users to cite their works when applicable.





