srun --container-image=openrlhf --nodelist=frl8a140 \
    --container-mounts=/lustre:/lustre \
    --pty -c90 --mem=400G --gpus=8 --container-mount-home  bash