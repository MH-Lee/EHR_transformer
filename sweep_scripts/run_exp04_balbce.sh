LOSS_TYPE=balanced_bce
PRETRAINED_TYPE=te3-large
DEVICE=cuda:1

python -u pretrained_main_sweeps3.py \
    --loss_type $LOSS_TYPE \
    --pretrained_type $PRETRAINED_TYPE \
    --device $DEVICE \
    --use_pretrained \
    --diag_freeze
