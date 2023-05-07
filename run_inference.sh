python3 inference.py --pretrained_model_path="../models/exp12/lightning_logs/version_0/checkpoints/epoch=4-step=3700.ckpt" \
    --test_parquet_path="/home/hovhannes/Downloads/cuisine_dataset_1/test/" \
    --device="cuda" \
    --batch_size=128 \
    --image_size=224