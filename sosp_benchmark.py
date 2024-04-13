import os
import subprocess
import itertools
import argparse

from sosp_parselog import _parselog


# interested_settings = [
#     # model size 7
#     dict(model_size=7, zero_stage=2, max_answer_len=128, offload=True, bs=128),
#     dict(model_size=7, zero_stage=2, max_answer_len=384, offload=True, bs=64),
#     dict(model_size=7, zero_stage=2, max_answer_len=896, offload=True, bs=32),
#     dict(model_size=7, zero_stage=3, max_answer_len=128, offload=True, bs=128),
#     dict(model_size=7, zero_stage=3, max_answer_len=384, offload=True, bs=64),
#     dict(model_size=7, zero_stage=3, max_answer_len=896, offload=True, bs=32),
#     dict(model_size=7, zero_stage=2, max_answer_len=128, offload=False, bs=128),
#     dict(model_size=7, zero_stage=2, max_answer_len=384, offload=False, bs=64),
#     dict(model_size=7, zero_stage=2, max_answer_len=896, offload=False, bs=32),
#     dict(model_size=7, zero_stage=3, max_answer_len=128, offload=False, bs=128),
#     dict(model_size=7, zero_stage=3, max_answer_len=384, offload=False, bs=64),
#     dict(model_size=7, zero_stage=3, max_answer_len=896, offload=False, bs=32),
#     # model size 13
#     dict(model_size=13, zero_stage=2, max_answer_len=128, offload=True, bs=64),
#     dict(model_size=13, zero_stage=2, max_answer_len=384, offload=True, bs=32),
#     dict(model_size=13, zero_stage=2, max_answer_len=896, offload=True, bs=16),
#     dict(model_size=13, zero_stage=2, max_answer_len=128, offload=False, bs=64),
#     dict(model_size=13, zero_stage=2, max_answer_len=384, offload=False, bs=32),
#     dict(model_size=13, zero_stage=2, max_answer_len=896, offload=False, bs=16),
#     dict(model_size=13, zero_stage=3, max_answer_len=128, offload=True, bs=64),
#     dict(model_size=13, zero_stage=3, max_answer_len=384, offload=True, bs=32),
#     dict(model_size=13, zero_stage=3, max_answer_len=896, offload=True, bs=16),
#     dict(model_size=13, zero_stage=3, max_answer_len=128, offload=False, bs=64),
#     dict(model_size=13, zero_stage=3, max_answer_len=384, offload=False, bs=32),
#     dict(model_size=13, zero_stage=3, max_answer_len=896, offload=False, bs=16),
#     # model size 34
#     dict(model_size=34, zero_stage=3, max_answer_len=128, offload=True, bs=32),
#     dict(model_size=34, zero_stage=3, max_answer_len=384, offload=True, bs=16),
#     dict(model_size=34, zero_stage=3, max_answer_len=896, offload=True, bs=8),
#     dict(model_size=34, zero_stage=3, max_answer_len=128, offload=False, bs=32),
#     dict(model_size=34, zero_stage=3, max_answer_len=384, offload=False, bs=16),
#     dict(model_size=34, zero_stage=3, max_answer_len=896, offload=False, bs=8),
#     # model size 70
#     dict(model_size=70, zero_stage=3, max_answer_len=128, offload=True, bs=16),
#     dict(model_size=70, zero_stage=3, max_answer_len=384, offload=True, bs=8),
#     dict(model_size=70, zero_stage=3, max_answer_len=896, offload=True, bs=4),
#     dict(model_size=70, zero_stage=3, max_answer_len=128, offload=False, bs=16),
#     dict(model_size=70, zero_stage=3, max_answer_len=384, offload=False, bs=8),
#     dict(model_size=70, zero_stage=3, max_answer_len=896, offload=False, bs=4),
# ]
interested_settings = []

def get_default_n_gpus(model_size: int):
    if model_size == 7:
        return 8
    elif model_size == 13:
        return 16
    elif model_size == 34:
        return 32
    elif model_size == 70:
        return 64
    else:
        raise NotImplementedError()


def build_default_sweep_settings(model_size: int, gpu_scale_factor: int):
    settings = []
    for model_size in [7, 13, 34, 70]:
        if model_size <= 13:
            zero_stages = [2]
        else:
            zero_stages = [3]
        default_n_gpus = get_default_n_gpus(model_size)
        n_gpus = default_n_gpus * gpu_scale_factor
        global_bs_seqlens = [(128, 896), (256, 384), (512, 128)] if default_n_gpus == n_gpus else [(256, 384)]
        for zero_stage, offload, (global_bs, genlen) in itertools.product(
            zero_stages, [False, True], global_bs_seqlens
        ):
            assert global_bs * (128 + genlen) == 2**17
            settings.append(
                dict(
                    model_size=model_size,
                    zero_stage=zero_stage,
                    max_answer_len=genlen,
                    bs=global_bs,
                    offload=offload,
                    n_gpus=n_gpus,
                )
            )
    return settings


def sweep(model_size: int, scale_actor: bool, scale_critic: bool, verbose_only: bool, gpu_scale_factor: int):
    assert scale_actor or scale_critic
    actor_size = model_size if scale_actor else 7
    critic_size = model_size if scale_critic else 7
    global interested_settings
    interested_settings = list(filter(lambda x: x["model_size"] == model_size, interested_settings))
    if len(interested_settings) == 0:
        settings = list(
            filter(lambda x: x["model_size"] == model_size, build_default_sweep_settings(model_size, gpu_scale_factor))
        )
        assert len(settings) > 0
        print(
            f">>>>>>>>>>>>>>>> No interested settings for actor {actor_size} critic {critic_size} found. Using default {len(settings)} settings. <<<<<<<<<<<<<<<<"
        )
    else:
        settings = interested_settings
        print(
            f">>>>>>>>>>>>>>>> Found interested settings for actor {actor_size} critic {critic_size}! Run interested {len(settings)} settings only. <<<<<<<<<<<<<<<<"
        )
    for setting in settings:
        default_n_gpus = get_default_n_gpus(max(actor_size, critic_size))
        n_gpus = setting["n_gpus"]

        zero_stage = setting["zero_stage"]
        max_answer_len = setting["max_answer_len"]
        bs = setting["bs"]
        offload = setting["offload"]
        if not args.force and _parselog(
            actor_size, critic_size, zero_stage, max_answer_len, bs, offload, gpu_scale_factor
        ):
            continue

        exp_name = f"or-a{actor_size}c{critic_size}s{max_answer_len // 100}g{bs // 100}"
        if offload:
            exp_name += "-of"
        if n_gpus != default_n_gpus:
            exp_name += f"x{gpu_scale_factor:1d}"

        trial_name = "1"
        logdir = f"/lustre/aigc/llm/logs/fw/{exp_name}/{trial_name}/"
        if os.path.exists(logdir):
            os.system(f"rm -rf {logdir}")
        cmd = (
            f"python3 slurm_ray_launch.py start "
            f" -e {exp_name} -f {trial_name} "
            f"--model_size {model_size} "
            f"--zero_stage {zero_stage} "
            f"--seqlen {max_answer_len} "
            f"--global_bs {bs} "
        )
        if n_gpus != default_n_gpus:
            cmd += f"--n_gpus {n_gpus} "
        if offload:
            cmd += "--offload "
        if scale_actor:
            cmd += "--scale_actor "
        if scale_critic:
            cmd += "--scale_critic "
        if verbose_only:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, choices=[7, 13, 34, 70], required=True, nargs="+")
    parser.add_argument("--scale_actor", "-a", action="store_true")
    parser.add_argument("--scale_critic", "-c", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--gpu_scale_factor", "-g", type=int, default=[1], nargs="+")
    parser.add_argument("--verbose_only", "-v", action="store_true")
    args = parser.parse_args()
    assert args.scale_actor or args.scale_critic
    for model_size, gpu_scale_factor in itertools.product(args.model_size, args.gpu_scale_factor):
        sweep(model_size, args.scale_actor, args.scale_critic, args.verbose_only, gpu_scale_factor)
