LOSS_TYPE=balanced_bce
DEVICE=cuda:0

python -u pretrained_main_sweeps3.py \
    --loss_type $LOSS_TYPE \
    --device $DEVICE
