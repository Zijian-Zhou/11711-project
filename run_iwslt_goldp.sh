#!/bin/bash
  
# #SBATCH --partition=cs
#SBATCH --job-name=ga_debug

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz1435@nyu.edu
#SBATCH --output=GOLD_debug_%A.out 
#SBATCH --error=GOLD_debug_%A.err

source activate fairseq

echo "SLURM_JOB_ID: " $SLURM_JOB_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID


TOTAL_NUM_UPDATES=150000
WARMUP_UPDATES=1000
UPDATE_FREQ=1

MLE_PATH="./MLE_models/iwslt.checkpoint.backup.pt.0" 

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en --save-dir checkpoints_iwslt_goldp_warmup4000 \
    --restore-file $MLE_PATH \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 3e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --p40 1 --use-is-obj 1 --save-interval-updates 10000 --keep-interval-updates 3 --policy-update-per-k-epoch 5000 --q-baseline -60.0 --iw-min 0.20 --reset-optimizer --trunc-min 0.1 --reward-type logp
