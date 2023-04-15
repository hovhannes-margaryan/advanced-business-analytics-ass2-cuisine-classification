python3 train.py --num_classes=20 \
 --pretrained_model_path="../models/exp1/lightning_logs/version_0/checkpoints/epoch=127-step=36224.ckpt" \
 --train_parquet_path="../train.parquet" \
 --validation_parquet_path="../validation.parquet" \
 --batch_size=128 \
 --device="cuda" \
 --image_size=224 \
 --max_epochs=500 \
 --model_checkpoint_path="../models/exp1" \
 --learning_rate=3e-4