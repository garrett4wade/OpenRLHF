import json
import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import argparse
from scipy.stats import t

pd.set_option("display.precision", 2)  # Set precision to 4 decimal places
np.set_printoptions(precision=2)  # Set precision to 4 decimal places


def t_score_ci(data):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Define confidence level (e.g., 95%)
    confidence_level = 0.95

    # Degrees of freedom (n-1 for a sample)
    degrees_of_freedom = len(data) - 1

    # Calculate the critical value based on the confidence level and degrees of freedom
    t_score = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = t_score * (std_dev / np.sqrt(len(data)))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound


benchmark_db = defaultdict(list)


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


def _parselog(
    actor_size: int,
    critic_size: int,
    zero_stage: int,
    seqlen: int,
    bs: int,
    offload: bool,
    gpu_scale_factor: int,
):
    exp_name = f"or-a{actor_size}c{critic_size}s{seqlen // 100}g{bs//100}"
    if offload:
        exp_name += "-of"
    if gpu_scale_factor != 1:
        exp_name += f"x{gpu_scale_factor:1d}"
    logpath = f"/lustre/aigc/llm/logs/fw/{exp_name}/1/rlhf.log"
    oom = False
    worker_max_mem = dict()
    oom_workers = set()
    time_records = []
    tflops_records = []
    thpt_records = []
    max_mem = 0.0
    gen_time_record = []
    train_time_record = []
    actor_train_time_record = []
    critic_train_time_record = []
    try:
        with open(logpath, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if "ray.exceptions.RayTaskError(OutOfMemoryError)" in line or "CUDA out of memory" in line:
                    oom = True
                    oom_workers.add(line.split()[1].strip())
                    # print(logpath)
                if "End-to-End" in line:
                    step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                    time_records.append(step_time)
                    tflops = float(line.split("TFLOPs: ")[1].split(",")[0])
                    tflops_records.append(tflops)
                    thpt = float(line.split(", Samples/sec: ")[1].split(",")[0])
                    thpt_records.append(thpt)
                if "Generation => Latency: " in line:
                    gen_time = float(line.split("Generation => Latency: ")[1].split("s,")[0])
                    gen_time_record.append(gen_time)
                if "Training   => Latency: " in line:
                    train_time = float(line.split("Training   => Latency: ")[1].split("s,")[0])
                    train_time_record.append(train_time)
                if "Compute Utilization - " in line:
                    mem = float(line.split("Used Memory - ")[1].split("MB,")[0])
                    if "pid" in line:
                        worker = line[6:].split(" ")[0]
                        worker_max_mem[worker] = max(worker_max_mem.get(worker, 0), mem)
                    max_mem = max(max_mem, mem)
                if "Actor forward time " in line:
                    fwd_time = float(line.split("Actor forward time ")[1].split("s")[0])
                    bwd_time = float(line.split(", backward time ")[1].split("s")[0])
                    actor_train_time_record.append(fwd_time + bwd_time)
                if "critic forward time " in line:
                    fwd_time = float(line.split("critic forward time ")[1].split("s")[0])
                    bwd_time = float(line.split(", backward time ")[1].split("s")[0])
                    critic_train_time_record.append(fwd_time + bwd_time)
    except FileNotFoundError:
        # print(f"File not found: {logpath}")
        return False
    time_records = time_records[2:]
    train_time_record = train_time_record[2:]
    gen_time_record = gen_time_record[2:]
    if not oom:
        if len(time_records) == 0 or len(tflops_records) == 0 or len(thpt_records) == 0 or max_mem == 0.0:
            return False
        avg_time = np.mean(time_records)
        var_time = np.var(time_records)
        min_time = np.min(time_records)
        max_time = np.max(time_records)
        cil, cih = t_score_ci(time_records)
        n_time = len(time_records)
        avg_train_time = np.mean(train_time_record)
        avg_gen_time = np.mean(gen_time_record)
        avg_actor_train_time = np.mean(actor_train_time_record)
        avg_critic_train_time = np.mean(critic_train_time_record)
        assert len(train_time_record) == n_time == len(gen_time_record)
        avg_tflops = np.mean(tflops_records)
        thpt = np.mean(thpt_records)
    else:
        avg_time = float("inf")
        avg_train_time = float("inf")
        avg_gen_time = avg_actor_train_time = avg_critic_train_time = float("inf")
        n_time = 0
        cil = cih = float("nan")
        min_time = float("nan")
        max_time = float("nan")
        var_time = float("nan")
        avg_tflops = -float("inf")
        thpt = -float("inf")
        max_mem = 0.0
    d = dict(
        a=actor_size,
        c=critic_size,
        s=seqlen,
        n_gpus=gpu_scale_factor * get_default_n_gpus(max(actor_size, critic_size)),
        z=zero_stage,
        # bs=bs,
        offload=offload,
        # OOM=oom,
        # Throughput=thpt,
        # MaxGPUMemory=max_mem,
        avg_t=avg_time,
        var_t=var_time,
        min_t=min_time,
        max_t=max_time,
        cil=cil,
        cih=cih,
        avg_gt=avg_gen_time,
        avg_tt=avg_train_time,
        avg_it=avg_time - avg_train_time - avg_gen_time,
        avg_mb_at=avg_actor_train_time,
        avg_mb_ct=avg_critic_train_time,
        pc_t=max(0, (avg_train_time - 4 * max(avg_actor_train_time, avg_critic_train_time))),
        n=n_time,
        log_path=logpath,
        # WorkerMaxMem=worker_max_mem,
        # OOMworker=oom_workers,
    )
    for k, v in d.items():
        benchmark_db[k].append(v)
    return True


def parselog(actor_size: int, critic_size: int, gpu_scale_factor: int):
    zero_stages = [2, 3]
    bszs_seqlens = [(128, 896), (256, 384), (512, 128)]
    offloads = [True, False]
    for (global_bs, seqlen), offload, zero_stage in itertools.product(bszs_seqlens, offloads, zero_stages):
        _parselog(actor_size, critic_size, zero_stage, seqlen, global_bs, offload, gpu_scale_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_size", "-a", type=int, default=[7, 13, 34, 70], choices=[7, 13, 34, 70], nargs="+"
    )
    parser.add_argument(
        "--critic_size", "-c", type=int, default=[7, 13, 34, 70], choices=[7, 13, 34, 70], nargs="+"
    )
    parser.add_argument("--gpu_scale_factor", "-g", default=[1, 2, 4, 8], nargs="+")
    parser.add_argument("--max", action="store_true")
    parser.add_argument("--dump_to_file", type=str, default=None)
    parser.add_argument("--no_print", action="store_true")
    args = parser.parse_args()
    for actor_size, critic_size, gpu_scale_factor in itertools.product(
        args.actor_size, args.critic_size, args.gpu_scale_factor
    ):
        parselog(actor_size, critic_size, gpu_scale_factor)
    df = pd.DataFrame(benchmark_db)
    if args.max:
        df = df.loc[df.groupby(["a", "c", "s", "n_gpus"])["avg_t"].idxmin()]
    if not args.no_print:
        print(df.to_string(index=False))
    if args.dump_to_file is not None:
        import pickle

        with open(args.dump_to_file, "wb") as f:
            pickle.dump(df, f)
