import itertools
import math
import os
import signal
import subprocess
from pathlib import Path
from typing import *

MODEL_SIZE_TO_PATH = {
    7: "/mnt/bs_fs/models/CodeLlama-7b-hf/",
    13: "/mnt/bs_fs/models/CodeLlama-13b-hf/",
    34: "/mnt/bs_fs/models/CodeLlama-34b-hf/",
    70: "/mnt/bs_fs/models/CodeLlama-70b-hf/",
}

MODEL_SIZE_TO_N_NODES_BAISC = {7: 2, 13: 4, 34: 8, 70: 16}
# Since openrlhf round-robinly assign actors to vllm engines,
# the number of vllm engines must be larger than the number of actors.
# With N GPUs, we have N//4 actors on N//4 GPUs and N//2 GPUs for vllm engines,
# so the minimum TP size should be 2.
MODEL_SIZE_TO_VLLM_TP_SIZE = {7: 2, 13: 2, 34: 2, 70: 4}


def build_cmd(
    model_size: int,
    bs: int,
    ctx: int,
    prompt_len: int,
    scale_both: bool,
    rollout_n_mbs: int,
    train_n_mbs: int,
    n_ppo_mbs=8,
):
    critic_size = model_size if scale_both else 7
    exp_name = "mlsys"
    trial_name = f"a{model_size}c{critic_size}b{bs}ct{ctx}p{prompt_len}nr{rollout_n_mbs}nt{train_n_mbs}"
    logfile = Path("/mnt/bs_fs/openrlhf-logs") / exp_name / trial_name / "output.log"

    actor_path = MODEL_SIZE_TO_PATH[model_size]
    critic_path = MODEL_SIZE_TO_PATH[critic_size]
    flags = get_common_flags(actor_path, critic_path)
    # Amend ctx & gen config
    if prompt_len >= ctx:
        return None
    flags += ["--prompt_max_len", prompt_len, "--generate_max_len", ctx - prompt_len]

    # Amend allocation
    n_nodes = MODEL_SIZE_TO_N_NODES_BAISC[model_size]
    n_vllm_gpus = n_nodes * 8 // 2
    vllm_tp_size = MODEL_SIZE_TO_VLLM_TP_SIZE[model_size]
    n_vllm_engines = n_vllm_gpus // vllm_tp_size

    actor_n_gpus = n_vllm_gpus // 2
    actor_n_nodes = (actor_n_gpus + 7) // 8
    actor_num_gpus_per_node = actor_n_gpus // actor_n_nodes

    for k in ["actor", "critic", "ref", "reward"]:
        flags += [
            f"--{k}_num_nodes",
            actor_n_nodes,
            f"--{k}_num_gpus_per_node",
            actor_num_gpus_per_node,
        ]
    flags += ["--vllm_num_engines", n_vllm_engines, "--vllm_tensor_parallel_size", vllm_tp_size]

    # Amend batch size
    flags += ["--rollout_batch_size", bs]
    if bs // rollout_n_mbs // actor_n_gpus < 1:
        return
    flags += ["--micro_rollout_batch_size", bs // rollout_n_mbs // actor_n_gpus]
    if bs // n_ppo_mbs < 1:
        return
    flags += ["--train_batch_size", bs // n_ppo_mbs]
    if bs // train_n_mbs // n_ppo_mbs // actor_n_gpus < 1:
        return
    flags += ["--micro_train_batch_size", bs // train_n_mbs // n_ppo_mbs // actor_n_gpus]
    flags += ["--critic_micro_train_batch_size", bs // train_n_mbs // n_ppo_mbs // actor_n_gpus]
    return " ".join(list(map(str, flags))), str(logfile)


def get_common_flags(actor_path, critic_path):
    return [
        "python3",
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--pretrain",
        actor_path,
        "--reward_pretrain",
        critic_path,
        "--zero_stage",
        3,
        "--normalize_reward",
        "--init_kl_coef",
        0.01,
        "--adam_offload",
        "--flash_attn",
        "--gradient_checkpointing",
        "--colocate_critic_reward",
        "--colocate_actor_ref",
        "--ref_reward_offload",
    ]


def log_stream_cmd(cmd, logfile):
    return f"stdbuf -oL {cmd} 2>&1 | tee -a {logfile}"


def run_interruptable_cmd_on_js_h100(cmd, nodelist, logfile, verbose=True):
    assert logfile is not None
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    os.system(f"touch {logfile}")
    cmd = f"/home/fw/openrlhf-bundle/rayrun {nodelist} {cmd}"
    if verbose:
        print(" Running command ".center(100, "=") + f"\n{cmd}\n" + "=" * 100 + "\n")
    try:
        pro = subprocess.Popen(
            log_stream_cmd(cmd, logfile),
            shell=True,
            preexec_fn=os.setsid,
        )
        pro.wait()
    except KeyboardInterrupt:
        for _ in range(3):
            pro.send_signal(signal.SIGINT)
        try:
            pro.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pro.terminate()
        try:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
