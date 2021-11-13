# GOLD implementation
 This repository an implementation of the GOLD algorithm from [# Text Generation by Learning from Demonstrations](https://arxiv.org/abs/2009.07839) based on [fairseq](https://github.com/pytorch/fairseq) and [the author's implementation](https://github.com/yzpang/gold-off-policy-text-gen-iclr21). Currently it only supports the IWSLT14 DE-EN dataset.

## Training
The script for training GOLD-p and GOLD-s are provided in ``` run_goldp.sh ``` and ``` run_golds.sh ```. The pretrained MLE model can be downloaded [here](https://drive.google.com/file/d/1dynOAM-EJ4ptfUeP8G5DR_vKbkcIo9tI/view?usp=sharing).

## Evaluation
To evaluate a trained model, run the following command:
```
python -W ignore [path-to-fairseq_cli/generate.py] data-bin/iwslt14.tokenized.de-en \
    --path [path-to-model-checkpoint.pt] \
    --batch-size 128 --beam 5 --remove-bpe --gen-subset test  > [path-to-save-to-file]
```
