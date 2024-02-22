import os
import subprocess
import itertools
import argparse

from sosp_parselog import _parselog

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", "-x", type=int, choices=[7, 13, 34, 70], required=True, nargs="+")
args = parser.parse_args()

interested_settings = []


def build_default_sweep_settings(model_size: int):
    settings = []
    seqlens = [256, 512, 1024]
    if model_size >= 34:
        zero_stages = [3]
    else:
        zero_stages = [2]
    for offload in [False, True]:
        if not offload:
            gen_batch_sizes = [48, 32, 24, 16, 8]
        else:
            gen_batch_sizes = list(reversed([16, 24, 32, 48, 64, 80]))
        for zero_stage, max_answer_len, gen_bs in itertools.product(zero_stages, seqlens, gen_batch_sizes):
            if not offload:
                if model_size <= 13 and gen_bs <= 4:
                    continue
                if max_answer_len == 256 and gen_bs <= 2:
                    continue
                if max_answer_len == 512 and gen_bs >= 8:
                    continue
                if max_answer_len == 1024 and gen_bs >= 8:
                    continue
            else:
                if model_size <= 13 and gen_bs <= 4:
                    continue
                if max_answer_len == 256 and gen_bs <= 8:
                    continue
                if max_answer_len == 512 and gen_bs >= 32:
                    continue
                if max_answer_len == 1024 and gen_bs >= 32:
                    continue
            settings.append(
                dict(
                    model_size=model_size,
                    zero_stage=zero_stage,
                    max_answer_len=max_answer_len,
                    bs=gen_bs,
                    offload=offload,
                )
            )
    return settings


def sweep(model_size: int):
    global interested_settings
    interested_settings = list(filter(lambda x: x["model_size"] == model_size, interested_settings))
    if len(interested_settings) == 0:
        settings = list(
            filter(lambda x: x["model_size"] == model_size, build_default_sweep_settings(model_size))
        )
        assert len(settings) > 0
        print(
            f">>>>>>>>>>>>>>>> No interested settings for model size {model_size} found. Using default {len(settings)} settings. <<<<<<<<<<<<<<<<"
        )
    else:
        settings = interested_settings
        print(
            f">>>>>>>>>>>>>>>> Found interested settings for model size {model_size}! Run interested {len(settings)} settings only. <<<<<<<<<<<<<<<<"
        )
    for setting in settings:
        assert model_size == setting["model_size"]
        zero_stage = setting["zero_stage"]
        max_answer_len = setting["max_answer_len"]
        bs = setting["bs"]
        offload = setting["offload"]
        if _parselog(model_size, zero_stage, max_answer_len, bs, offload):
            continue

        exp_name = f"or-a{model_size}s{max_answer_len // 100}g{bs}"
        if offload:
            exp_name += "-of"
        trial_name = "1"
        cmd = (
            f"python3 slurm_ray_launch.py start "
            f" -e {exp_name} -f {trial_name} "
            f"--model_size {model_size} "
            f"--zero_stage {zero_stage} "
            f"--seqlen {max_answer_len} "
            f"--per_device_bs {bs} "
        )
        if offload:
            cmd += "--offload "
        os.system(cmd)
        # print(cmd)


if __name__ == "__main__":
    for model_size in args.model_size:
        sweep(model_size)
