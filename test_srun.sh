srun --container-image=llm/llm-openrlhf --exclude=QH-com02,QH-com03,QH-com29,QH-com35 \
    --container-mounts=/lustre:/lustre,/home/fw/sosp-workspace/OpenRLHF/openrlhf-vllm:/vllm \
    --pty -c99 --mem=900G --gpus=8 --container-mount-home  bash