srun --container-image=llm/llm-openrlhf --exclude=QH-com29,QH-com35 \
    --container-mounts=/lustre:/lustre \
    --pty -c99 --mem=900G --gpus=8 --container-mount-home  bash