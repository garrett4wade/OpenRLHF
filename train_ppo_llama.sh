deepspeed train_ppo.py \
    --pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf \
    --reward_pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf \
    --logging_steps 1 \
    --micro_train_batch_size 32 \
    --train_batch_size 1024 \
    --micro_rollout_batch_size 64 \
    --rollout_batch_size 512 \
    --prompt_max_len 256 \
    --generate_max_len 256 \
    --zero_stage 3 \
    --bf16 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing

# if [[ ${1} != "slurm" ]]; then
#     export PATH=$HOME/.local/bin/:$PATH
#     deepspeed $training_commands
# fi
