import json
import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import argparse

benchmark_db = defaultdict(list)


def _parselog(
    actor_size: int,
    critic_size: int,
    zero_stage: int,
    seqlen: int,
    bs: int,
    offload: bool,
):
    exp_name = f"or-a{actor_size}c{critic_size}s{seqlen // 100}g{bs//100}"
    if offload:
        exp_name += "-of"
    logpath = f"/lustre/aigc/llm/logs/fw/{exp_name}/1/rlhf.log"
    oom = False
    worker_max_mem = dict()
    oom_workers = set()
    time_records = []
    tflops_records = []
    thpt_records = []
    max_mem = 0.0
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
                if "Compute Utilization - " in line:
                    mem = float(line.split("Used Memory - ")[1].split("MB,")[0])
                    if "pid" in line:
                        worker = line[6:].split(" ")[0]
                        worker_max_mem[worker] = max(worker_max_mem.get(worker, 0), mem)
                    max_mem = max(max_mem, mem)
    except FileNotFoundError:
        # print(f"File not found: {logpath}")
        return False
    if not oom:
        if len(time_records) == 0 or len(tflops_records) == 0 or len(thpt_records) == 0 or max_mem == 0.0:
            return False
        avg_time = np.mean(time_records)
        avg_tflops = np.mean(tflops_records)
        thpt = np.mean(thpt_records)
    else:
        avg_time = float("inf")
        avg_tflops = -float("inf")
        thpt = -float("inf")
        max_mem = 0.0
    d = dict(
        actor_size=actor_size,
        critic_size=critic_size,
        zero_stage=zero_stage,
        seqlen=seqlen,
        bs=bs,
        offload=offload,
        OOM=oom,
        Throughput=thpt,
        MaxGPUMemory=max_mem,
        avg_time=avg_time,
        # WorkerMaxMem=worker_max_mem,
        # OOMworker=oom_workers,
        gpu_scale_factor=1,
    )
    for k, v in d.items():
        benchmark_db[k].append(v)
    return True


def parselog(actor_size: int, critic_size: int):
    zero_stages = [2, 3]
    bszs_seqlens = [(128, 896), (256, 384), (512, 128)]
    offloads = [True, False]
    for (global_bs, seqlen), offload, zero_stage in itertools.product(bszs_seqlens, offloads, zero_stages):
        _parselog(actor_size, critic_size, zero_stage, seqlen, global_bs, offload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_size", "-a", type=int, default=[7, 13, 34, 70], choices=[7, 13, 34, 70], nargs="+"
    )
    parser.add_argument(
        "--critic_size", "-c", type=int, default=[7, 13, 34, 70], choices=[7, 13, 34, 70], nargs="+"
    )
    parser.add_argument("--max", action="store_true")
    parser.add_argument("--dump_to_file", type=str, default=None)
    parser.add_argument("--no_print", action="store_true")
    args = parser.parse_args()
    for actor_size, critic_size in itertools.product(args.actor_size, args.critic_size):
        parselog(actor_size, critic_size)
    df = pd.DataFrame(benchmark_db)
    if args.max:
        df = df.loc[df.groupby(["actor_size", "critic_size", "seqlen"])["Throughput"].idxmax()]
    if not args.no_print:
        print(df.to_string(index=False))
    if args.dump_to_file is not None:
        import pickle

        with open(args.dump_to_file, "wb") as f:
            pickle.dump(df, f)
