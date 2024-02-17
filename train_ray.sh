ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/home/fw/sosp-workspace/OpenRLHF"}' \
    -- python3 train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf \
    --reward_pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf \
    --micro_train_batch_size 32 \
    --critic_micro_train_batch_size 64 \
    --train_batch_size 512 \
    --micro_rollout_batch_size 128 \
    --rollout_batch_size 512 \
    --prompt_max_len 256 \
    --generate_max_len 256 \
    --zero_stage 2 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing