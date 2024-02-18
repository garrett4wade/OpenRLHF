import argparse
import dataclasses
import getpass
import os
import random
import socket
import re
import subprocess
import time
from typing import Dict, List, Optional


import base.cluster as cluster
import base.logging as logging
import base.name_resolve
import base.names
import scheduler.client
from base.constants import (
    DATASET_CACHE_PATH,
    LOG_ROOT,
    PYTORCH_KERNEL_CACHE_PATH,
    TORCH_EXTENSIONS_DIR,
    TRITON_CACHE_PATH,
)

logger = logging.getLogger("main", "system")
WORKER_TYPE = "default-ray-cluster"


def get_random_port(min_port=1024, max_port=49151):
    while True:
        port = random.randint(min_port, max_port)
        if is_port_available(port):
            return port


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


class RayController:
    """A controller that uses Ray to manage workers.

    It uses the basic Controller to configure workers.
    Besides, it launchs all remote workers using Ray,
    instead of submitting them to the scheduelr.
    """

    def __init__(
        self,
        experiment_name,
        trial_name,
        local_mode: bool,
        driver_cmd: str,
        ray_cluster_count: int = None,
    ):
        # base controller will be lazier initialized when launching workers.
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name

        self.__local_mode = local_mode
        self.__ray_cluster_count = ray_cluster_count

        self.__driver_cmd = driver_cmd

    def start(self):
        if not self.__local_mode:
            logger.info(f"Waiting for {self.__ray_cluster_count} Ray cluster jobs to start...")
            for idx in range(self.__ray_cluster_count):
                try:
                    base.name_resolve.wait(
                        base.names.ray_cluster(
                            self.__experiment_name, self.__trial_name, f"{WORKER_TYPE}/{idx}"
                        ),
                        timeout=300,
                    )
                except TimeoutError:
                    raise RuntimeError(f"Timeout waiting for Ray cluster node {WORKER_TYPE}/{idx} to start.")
            logger.info("Ray cluster started.")

        try:
            ray_head_addr = base.name_resolve.wait(
                base.names.ray_cluster(self.__experiment_name, self.__trial_name, "address"), timeout=300
            )
        except TimeoutError:
            raise RuntimeError("Timeout waiting for ray cluster head address.")

        logger.info("Ray initialized! Ready to run workers.")

        # Create Ray job command.
        runtime_env_json = '{"working_dir": "/home/fw/sosp-workspace/OpenRLHF"}'
        ray_job_cmd = f"ray job submit --no-wait --address=\"{ray_head_addr}\" --runtime-env-json='{runtime_env_json}' -- {self.__driver_cmd}"
        logger.info(f"Ready to run Ray job with command:\n{ray_job_cmd}")

        # Obtain Ray job id. Used for redirecting log file.
        ray_job_output = subprocess.check_output(ray_job_cmd, shell=True).decode("utf-8")
        job_id = None
        for line in ray_job_output.split("\n"):
            if "submitted successfully" in line:
                job_id = line.split("'")[1]
        if job_id is None:
            raise RuntimeError("Job submission failed.")

        # Redirect logfile.
        logfile = os.path.join(
            LOG_ROOT,
            self.__experiment_name,
            self.__trial_name,
            "session_latest/logs",
            f"job-driver-{job_id}.log",
        )
        logger.info(f">>>>> Check the Ray experiment log with: <<<<<\n\ttail -f {logfile} ")

        try:
            while True:
                output = subprocess.check_output(f"ray job status {job_id}", shell=True).decode("utf-8")
                if "RUNNING" not in output:
                    raise RuntimeError(f"Job {job_id} status is not running. Ray output:\n{output}")
                time.sleep(10)
        except Exception as e:
            logger.info(f"Stop the Ray job due to exception:\n{e}")
            subprocess.check_output(f"ray job stop {job_id}", shell=True)
        finally:
            self.shutdown()

    def shutdown(self):
        ray_exiting_name = base.names.ray_cluster(self.__experiment_name, self.__trial_name, "exiting")
        base.name_resolve.add(ray_exiting_name, value="1", delete_on_exit=True)


CONTROLLER_TIME_LIMIT = None
_LLM_ENVVARS = {
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "TOKENIZERS_PARALLELISM": "true",
    "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
    "RAY_DEDUP_LOGS": "0",  # disable ray log deduplication
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUSERBASE": "/nonsense",
}
for k, v in _LLM_ENVVARS.items():
    os.environ[k] = v


@dataclasses.dataclass
class Scheduling:
    count: int
    cpu: int = 16
    gpu: int = 8
    mem: int = 100000
    gpu_type: str = "tesla"
    node_type: str = None
    nodelist: str = None
    exclude: str = "QH-com29,QH-com35"
    container_image: str = "llm/llm-openrlhf"
    env_vars: Dict[str, str] = dataclasses.field(default_factory=lambda: _LLM_ENVVARS)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format


def driver_cmd(args):
    ngpus, device_partition, nodelist = get_ngpus_and_nodelist_from_model_size(
        args.model_size, args.colocate_actor_critic, args.colocate_ref_rew
    )
    assert all(g <= 8 or g % 8 == 0 for g in device_partition)
    n_actor_gpus = device_partition[0]
    n_critic_gpus = device_partition[1]
    n_ref_gpus = device_partition[2]
    n_rew_gpus = device_partition[3]
    if args.colocate_actor_critic:
        assert n_actor_gpus == n_critic_gpus
        actor_critic_gpus = n_actor_gpus
    else:
        actor_critic_gpus = n_actor_gpus + n_critic_gpus
    if args.colocate_ref_rew:
        assert n_ref_gpus == n_rew_gpus
        ref_rew_gpus = n_ref_gpus
    else:
        ref_rew_gpus = n_ref_gpus + n_rew_gpus
    assert ngpus == actor_critic_gpus + ref_rew_gpus
    cmd = ["python3", "train_ppo_ray.py"]
    flags = [
        f"--ref_num_nodes {n_ref_gpus // 8 if n_ref_gpus > 8 else 1}",
        f"--ref_num_gpus_per_node {8 if n_ref_gpus > 8 else n_ref_gpus}",
        f"--reward_num_nodes {n_rew_gpus // 8 if n_rew_gpus > 8 else 1}",
        f"--reward_num_gpus_per_node {8 if n_rew_gpus > 8 else n_rew_gpus}",
        f"--actor_num_nodes {n_actor_gpus // 8 if n_actor_gpus > 8 else 1}",
        f"--actor_num_gpus_per_node {8 if n_actor_gpus > 8 else n_actor_gpus}",
        f"--critic_num_nodes {n_critic_gpus // 8 if n_critic_gpus > 8 else 1}",
        f"--train_batch_size {4 * n_actor_gpus * args.per_device_train_batch_size}",
        f"--micro_train_batch_size {args.per_device_train_batch_size}",
        f"--critic_micro_train_batch_size {args.per_device_train_batch_size * n_actor_gpus // n_critic_gpus}",
        f"--rollout_batch_size {n_actor_gpus * args.per_device_gen_batch_size}",
        f"--micro_rollout_batch_size {args.per_device_gen_batch_size}",
        f"--max_epochs 1",
        f"--prompt_max_len 256",
        f"--generate_max_len {args.seqlen}",
        f"--zero_stage {args.zero_stage}",
        "--normalize_reward",
        "--actor_init_on_gpu",
        "--flash_attn",
        "--gradient_checkpointing",
    ]
    if args.offload:
        flags.append("--adam_offload")
    model_path = get_path_from_model_size(args.model_size)
    flags.extend(
        [
            f"--pretrain {model_path}",
            f"--reward_pretrain {model_path}",
        ]
    )
    if args.zero_stage == 3:
        flags.append("--bf16")
    if args.vllm_num_engines > 0:
        assert args.model_size > 7
        flags.append(f"--vllm_num_engines {args.vllm_num_engines}")
        flags.append(
            f"--vllm_tensor_parallel_size {2 if args.model_size ==13 else 4 if args.model_size==34 else 8}"
        )
    return " ".join(cmd + flags)


def control_cmd(expr_name, trial_name, local_mode):
    cmd = f"python3 remote.py controller -e {expr_name} -f {trial_name}"
    if local_mode:
        cmd += " --local_mode"
    return cmd


def ray_cluster_cmd(expr_name, trial_name, worker_type):
    flags = [f"-e {expr_name}", f"-f {trial_name}", f"-w {worker_type}"]
    return (
        f"python3 remote.py ray -i {{jobstep_id}} -g {{n_jobsteps}} -r {{worker_submission_index}} "
        f"-p {{wprocs_per_jobstep}} -j {{wprocs_in_job}} -o {{wproc_offset}} {' '.join(flags)}"
    )


def main_ray_driver(args):
    ray_port = args.ray_port
    # launch ray cluster head
    ray_flags = [
        f"--port={ray_port}",
        "--head",
        f"--temp-dir={os.path.join(LOG_ROOT, args.experiment_name, args.trial_name)}",
    ]
    if args.mode == "slurm":
        ray_flags.extend(
            [
                f"--num-cpus=0",
                f"--num-gpus=0",
            ]
        )
    cmd = f"ray start {' '.join(ray_flags)}"
    output = subprocess.check_output(cmd, shell=True).decode("ascii")
    logger.info("Successfully launched ray cluster head.")

    pattern = r"ray start --address='(\d+\.\d+\.\d+\.\d+:\d+)'"
    match = re.search(pattern, output)
    if match:
        addr = match.group(1)
        logger.info("Found ray address: '%s'", addr)
    else:
        raise RuntimeError(f"Address not found in ray start output: {output}.")
    ray_addr_name = base.names.ray_cluster(args.experiment_name, args.trial_name, "address")
    base.name_resolve.add(ray_addr_name, addr, delete_on_exit=True, keepalive_ttl=500)

    # For slurm model, launch the Ray cluster via slurm command.
    ngpus, _, _ = get_ngpus_and_nodelist_from_model_size(
        args.model_size, args.colocate_actor_critic, args.colocate_ref_rew
    )

    controller = RayController(
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
        local_mode=args.mode == "local",
        ray_cluster_count=None if args.mode == "local" else ngpus,
        driver_cmd=driver_cmd(args),
    )
    try:
        controller.start()
    except (KeyboardInterrupt, Exception) as e:
        subprocess.check_output(f"ray stop", shell=True)
        raise e
    subprocess.check_output(f"ray stop", shell=True)


def main_start(args):
    image_name = "llm/llm-openrlhf"
    args.ray_port = ray_port = get_random_port()

    trial_name = args.trial_name
    expr_name = args.experiment_name
    sched = scheduler.client.make(mode=args.mode, expr_name=expr_name, trial_name=trial_name)

    base_environs = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
    }

    logger.info(f"Resetting name resolving repo...")

    try:
        base.name_resolve.clear_subtree(
            base.names.trial_root(experiment_name=args.experiment_name, trial_name=args.trial_name)
        )
    except Exception as e:
        logger.warning(f"Resetting name resolving repo failed.")
        raise e
    logger.info(f"Resetting name resolving repo... Done.")

    if args.mode == "local":
        main_ray_driver(args)
    else:
        remote_driver_cmd = " ".join(
            [
                "python3",
                "slurm_ray_launch.py",
                "driver",
                f"-e {args.experiment_name}",
                f"-f {args.trial_name}",
                f"--model_size {args.model_size}",
                f"--mode {args.mode}",
                f"--vllm_num_engines {args.vllm_num_engines}",
                f"--per_device_train_batch_size {args.per_device_train_batch_size}",
                f"--per_device_gen_batch_size {args.per_device_gen_batch_size}",
                f"--zero_stage {args.zero_stage}",
                f"--offload" if args.offload else "",
                f"--seqlen {args.seqlen}",
                "--colocate_ref_rew" if args.colocate_ref_rew else "",
                "--colocate_actor_critic" if args.colocate_actor_critic else "",
            ]
        )
        sched.submit_array(
            worker_type="ctl",
            cmd=remote_driver_cmd,
            count=1,
            cpu=4,
            gpu=0,
            mem=32 * 1024,
            env_vars=base_environs,
            container_image=image_name,
            time_limit=CONTROLLER_TIME_LIMIT,
        )
        ngpus, _, _ = get_ngpus_and_nodelist_from_model_size(
            args.model_size, args.colocate_actor_critic, args.colocate_ref_rew
        )
        sch_cfg = Scheduling(count=ngpus)
        job_environs = {**base_environs, **sch_cfg.env_vars}
        cmd = ray_cluster_cmd(
            expr_name,
            trial_name,
            worker_type=WORKER_TYPE,
        )

        logger.debug(f"Scheduling Ray cluster...")

        nodelist = sch_cfg.nodelist
        exclude = sch_cfg.exclude
        node_type = sch_cfg.node_type
        container_image = image_name or sch_cfg.container_image

        sched.submit_array(
            worker_type=WORKER_TYPE,
            cmd=cmd,
            count=sch_cfg.count,
            cpu=sch_cfg.cpu,
            gpu=sch_cfg.gpu,
            gpu_type=sch_cfg.gpu_type,
            mem=sch_cfg.mem,
            container_image=container_image,
            node_type=node_type,
            nodelist=nodelist,
            exclude=exclude,
            env_vars=job_environs,
            hostfile=True,
            multiprog=True,
            begin=sch_cfg.begin,
            deadline=sch_cfg.deadline,
            time_limit=sch_cfg.time_limit,
        )
        try:
            sched.wait()
        except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
            sched.stop_all()
            raise e


def get_path_from_model_size(model_size: int):
    if model_size == 7:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
    elif model_size == 13:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
    elif model_size == 34:
        model_path = "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf"
    elif model_size == 70:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-70b-hf"
    else:
        raise NotImplementedError()
    return model_path


def get_ngpus_and_nodelist_from_model_size(
    model_size: int, colocate_actor_critic: bool, colocate_ref_rew: bool
):
    device_partition = None
    if model_size in [7]:
        if colocate_actor_critic and colocate_ref_rew:
            device_partition = (6, 6, 2, 2)
        elif colocate_actor_critic:
            device_partition = (6, 6, 1, 1)
        elif colocate_ref_rew:
            device_partition = (4, 2, 2, 2)
        else:
            device_partition = (4, 2, 1, 1)
        ngpus, nodelist = 8, "QH-com01"
    elif model_size == 13:
        ngpus, nodelist = 16, "QH-com[02-03]"
    elif model_size in [34]:
        ngpus, nodelist = 32, "QH-com[04-06,09]"
    elif model_size == 70:
        ngpus, nodelist = 64, "QH-com[36-43]"
    return ngpus, device_partition, nodelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="launch ray cluster and run workers")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--mode", default="slurm", choices=["local", "slurm"])

    subparser.add_argument("--model_size", type=int, choices=[7, 13, 34, 70], required=True)
    subparser.add_argument("--vllm_num_engines", type=int, default=0)
    subparser.add_argument("--colocate_ref_rew", action="store_true")
    subparser.add_argument("--colocate_actor_critic", action="store_true")
    subparser.add_argument("--per_device_train_batch_size", type=int, default=4)
    subparser.add_argument("--per_device_gen_batch_size", type=int, default=4)
    subparser.add_argument("--zero_stage", type=int, default=2)
    subparser.add_argument("--offload", action="store_true")
    subparser.add_argument("--seqlen", type=int, default=256)
    subparser.set_defaults(func=main_start)

    subparser = subparsers.add_parser("driver", help="launch ray cluster and run workers")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--mode", default="slurm", choices=["local", "slurm"])

    subparser.add_argument("--model_size", type=int, choices=[7, 13, 34, 70], required=True)
    subparser.add_argument("--vllm_num_engines", type=int, default=0)
    subparser.add_argument("--colocate_ref_rew", action="store_true")
    subparser.add_argument("--colocate_actor_critic", action="store_true")
    subparser.add_argument("--per_device_train_batch_size", type=int, default=4)
    subparser.add_argument("--per_device_gen_batch_size", type=int, default=4)
    subparser.add_argument("--zero_stage", type=int, default=2)
    subparser.add_argument("--offload", action="store_true")
    subparser.add_argument("--seqlen", type=int, default=256)
    subparser.set_defaults(func=main_ray_driver)

    args = parser.parse_args()
    args.func(args)
