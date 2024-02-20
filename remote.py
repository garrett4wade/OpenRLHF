from typing import *
import argparse
import multiprocessing
import os
import pickle
import re
import socket
import copy
import subprocess

import torch
import ray
import ray.util.queue as rq
import base.names as names

multiprocessing.set_start_method("spawn", force=True)

from base.constants import QUICKSTART_EXPR_CACHE_PATH
import base.gpu_utils
import base.logging as logging
import base.name_resolve
import base.names

RAY_HEAD_WAIT_TIME = 500
logger = logging.getLogger("Main-Workers")


def main_ray(args):
    worker_type = "default-ray-cluster"
    ray_addr_name = base.names.ray_cluster(args.experiment_name, args.trial_name, "address")
    try:
        address = base.name_resolve.wait(ray_addr_name, timeout=RAY_HEAD_WAIT_TIME)
    except TimeoutError:
        raise TimeoutError("Timeout waiting for ray cluster head address.")
    ray_flags = [f"--address={address}"]

    cmd = f"ray start {' '.join(ray_flags)}"
    _ = subprocess.check_output(cmd, shell=True).decode("ascii")
    logger.info(f"Successfully launched nodes for {worker_type} in Ray cluster.")

    host_ip = socket.gethostbyname(socket.gethostname())
    base.name_resolve.add(
        base.names.ray_cluster(args.experiment_name, args.trial_name, f"{worker_type}/{args.jobstep_id}"),
        host_ip,
        delete_on_exit=True,
        keepalive_ttl=300,
    )

    while True:
        try:
            ray_exiting_name = base.names.ray_cluster(args.experiment_name, args.trial_name, "exiting")
            try:
                base.name_resolve.wait(ray_exiting_name, timeout=10)
                break
            except TimeoutError:
                pass
        except KeyboardInterrupt:
            break

    subprocess.check_output(f"ray stop", shell=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("ray", help="launch ray cluster write ray address to name_resolve")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument(
        "--jobstep_id", "-i", type=int, required=True, help="jobstep/task ID in a slurm job."
    )
    subparser.add_argument(
        "--n_jobsteps", "-g", type=int, required=True, help="`--ntasks` of `srun`, aka SLURM_NPROCS."
    )
    subparser.add_argument(
        "--worker_submission_index",
        "-r",
        type=int,
        required=True,
        help="Submission index to slurm for this worker. Used for locating job name and logs.",
    )
    subparser.add_argument(
        "--wprocs_per_jobstep",
        "-p",
        type=int,
        required=True,
        help="Number of worker processes launched by multiprocessing in this script.",
    )
    subparser.add_argument(
        "--wprocs_in_job", "-j", type=int, required=True, help="Number of worker processes in this slurm job."
    )
    subparser.add_argument(
        "--wproc_offset",
        "-o",
        type=int,
        required=True,
        help="Offset of worker processes of this slurm job. "
        "For example, we may allocate 4 type `A` workers with 1 GPU each and 2 with 0.5 GPU each. "
        "This launches 2 jobs, the former with 4 job steps and the latter with 2 job steps. "
        "The offset is 0 for the 1st job and 4 for the 2nd job.",
    )
    subparser.set_defaults(func=main_ray)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
