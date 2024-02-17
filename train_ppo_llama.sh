deepspeed train_ppo.py \
    --pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf \
    --reward_pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf \
    --logging_steps 1 \
    --micro_train_batch_size 11 \
    --train_batch_size 352 \
    --micro_rollout_batch_size 44 \
    --rollout_batch_size 352 \
    --prompt_max_len 256 \
    --generate_max_len 256 \
    --zero_stage 2 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing

# if [[ ${1} != "slurm" ]]; then
#     export PATH=$HOME/.local/bin/:$PATH
#     deepspeed $training_commands
# fi
