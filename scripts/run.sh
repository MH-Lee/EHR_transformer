BATCH_SIZE=256
LEARNING_RATE=5e-4
MASK_PROB=0.15
DIM_FEEDFORWARD=2048
LOSS_TYPE=balanced_bce
EMB_DIM=256
EPOCHS=100
NUM_HEADS=8
NUM_LAYERS=4
NUM_CLASSES=10
LABEL_TYPE=bot
ATTN_DROPOUT=0.3
DROPOUT=0.2
MLM_LAMBDA=0.35
GAMMA=0.0
ALPHA=0.2
POOL_TYPE=mean
EXP_NUM=1
DEVICE=cuda:0

python -u pretrained_main.py \
            --batch_size $BATCH_SIZE \
            --max_epoch $EPOCHS \
            --lr $LEARNING_RATE \
            --embed_dim $EMB_DIM \
            --mask_prob $MASK_PROB \
            --ffn_dim $DIM_FEEDFORWARD \
            --loss_type $LOSS_TYPE \
            --num_heads $NUM_HEADS \
            --num_layers $NUM_LAYERS \
            --num_classes $NUM_CLASSES \
            --label_type $LABEL_TYPE \
            --attn_dropout $ATTN_DROPOUT \
            --dropout $DROPOUT \
            --gamma $GAMMA \
            --alpha $ALPHA \
            --exp_num $EXP_NUM \
            --device $DEVICE \
            --mlm_lambda $MLM_LAMBDA \
            --pool_type $POOL_TYPE \
            --use_wandb