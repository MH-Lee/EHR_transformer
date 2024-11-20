BATCH_SIZE=256
LEARNING_RATE=5e-4
MASK_PROB=0.3
DIM_FEEDFORWARD=2048
LOSS_TYPE=balanced_bce
PRETRAINED_TYPE=te3-small
EMB_DIM=512
EPOCHS=100
NUM_HEADS=8
NUM_LAYERS=4
ATTN_DROPOUT=0.3
DROPOUT=0.2
MLM_LAMBDA=0.15
GAMMA=2.0
EXP_NUM=3
DEVICE=cuda:1


python -u pretrained_main.py \
        --batch_size $BATCH_SIZE \
        --max_epoch $EPOCHS \
        --lr $LEARNING_RATE \
        --embed_dim $EMB_DIM \
        --mask_prob $MASK_PROB \
        --ffn_dim $DIM_FEEDFORWARD \
        --loss_type $LOSS_TYPE \
        --pretrained_type $PRETRAINED_TYPE \
        --num_heads $NUM_HEADS \
        --num_layers $NUM_LAYERS \
        --attn_dropout $ATTN_DROPOUT \
        --dropout $DROPOUT \
        --gamma $GAMMA \
        --exp_num $EXP_NUM \
        --device $DEVICE \
        --mlm_lambda $MLM_LAMBDA \
        --use_pretrained \
        --use_wandb

EXP_NUM=4

python -u pretrained_main.py \
        --batch_size $BATCH_SIZE \
        --max_epoch $EPOCHS \
        --lr $LEARNING_RATE \
        --embed_dim $EMB_DIM \
        --mask_prob $MASK_PROB \
        --ffn_dim $DIM_FEEDFORWARD \
        --loss_type $LOSS_TYPE \
        --pretrained_type $PRETRAINED_TYPE \
        --num_heads $NUM_HEADS \
        --num_layers $NUM_LAYERS \
        --attn_dropout $ATTN_DROPOUT \
        --dropout $DROPOUT \
        --gamma $GAMMA \
        --exp_num $EXP_NUM \
        --device $DEVICE \
        --mlm_lambda $MLM_LAMBDA \
        --use_pretrained \
        --diag_freeze \
        --use_wandb