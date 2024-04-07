import ray
import argparse
import dataclasses
import getpass
import os
import random
import socket
from datetime import datetime
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
        job_submission_addr: str,
        ray_cluster_count: int = None,
    ):
        # base controller will be lazier initialized when launching workers.
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name

        self.__local_mode = local_mode
        self.__ray_cluster_count = ray_cluster_count

        self.__driver_cmd = driver_cmd

        self.__job_submission_addr = job_submission_addr

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

        logger.info("Ray initialized! Ready to run workers.")

        # Create Ray job command.
        # ray.init()
        # train_args = build_train_args(args)

        # from train_ppo_ray import train

        # train(train_args)

        # self.shutdown()

        job_id = "rlhf"
        runtime_env_json = '{"working_dir": "/home/fw/sosp-workspace/OpenRLHF"}'
        ray_job_cmd = (
            f"ray job submit --no-wait "
            # f"--entrypoint-num-gpus=1 --entrypoint-num-cpus=2 "
            f"--runtime-env-json='{runtime_env_json}' --address={self.__job_submission_addr} --submission-id {job_id} -- {self.__driver_cmd}"
        )
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
            "/tmp/ray",
            self.__experiment_name,
            self.__trial_name,
            "session_latest/logs",
            f"job-driver-{job_id}.log",
        )
        new_logfile = os.path.join(LOG_ROOT, self.__experiment_name, self.__trial_name, f"{job_id}.log")
        os.makedirs(os.path.dirname(new_logfile), exist_ok=True, mode=0o775)
        logger.info(f"Local Ray job log path: {logfile}")

        retarget_proc = None
        try:
            while True:
                if retarget_proc is None and os.path.exists(logfile):
                    retarget_proc = subprocess.Popen(f"tail -f {logfile} >> {new_logfile}", shell=True)
                    logger.info(f">>>>> Check the Ray experiment log with: <<<<<\n\ttail -f {new_logfile} ")
                output = subprocess.check_output(f"ray job status {job_id}", shell=True).decode("utf-8")
                if "RUNNING" not in output:
                    raise RuntimeError(f"Job {job_id} status is not running. Ray output:\n{output}")
                time.sleep(10)
        except Exception as e:
            logger.info(f"Stop the Ray job due to exception:\n{e}")
            subprocess.check_output(f"ray job stop {job_id}", shell=True)
        finally:
            retarget_proc.kill()
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
    "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
    "PYTHONUSERBASE": "/nonsense",
}
for k, v in _LLM_ENVVARS.items():
    os.environ[k] = v


@dataclasses.dataclass
class Scheduling:
    count: int
    cpu: int = 8
    gpu: int = 1
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
    ngpus, device_partition, nodelist, colocate_actor_critic, colocate_ref_reward = (
        get_ngpus_and_nodelist_from_model_size(
            args.model_size,
            scale_actor=args.scale_actor,
            scale_critic=args.scale_critic,
        )
    )
    # assert all(g <= 8 or g % 8 == 0 for g in device_partition)
    n_actor_gpus = device_partition[0]
    n_critic_gpus = device_partition[1]
    n_ref_gpus = device_partition[2]
    n_rew_gpus = device_partition[3]
    n_vllm_engines_gpus = device_partition[4]
    if colocate_actor_critic:
        assert n_actor_gpus == n_critic_gpus
        actor_critic_gpus = n_actor_gpus
    else:
        actor_critic_gpus = n_actor_gpus + n_critic_gpus
    if colocate_ref_reward:
        assert n_ref_gpus == n_rew_gpus
        ref_rew_gpus = n_ref_gpus
    else:
        ref_rew_gpus = n_ref_gpus + n_rew_gpus
    cmd = ["python3", "train_ppo_ray.py"]
    actor_per_device_bs = args.global_bs // n_actor_gpus
    critic_per_device_bs = args.global_bs // n_critic_gpus
    assert actor_per_device_bs // 4 > 0
    assert critic_per_device_bs // 4 > 0
    flags = [
        f"--ref_num_nodes {n_ref_gpus // 8 if n_ref_gpus > 8 else 1}",
        f"--ref_num_gpus_per_node {8 if n_ref_gpus > 8 else n_ref_gpus}",
        f"--reward_num_nodes {n_rew_gpus // 8 if n_rew_gpus > 8 else 1}",
        f"--reward_num_gpus_per_node {8 if n_rew_gpus > 8 else n_rew_gpus}",
        f"--actor_num_nodes {n_actor_gpus // 8 if n_actor_gpus > 8 else 1}",
        f"--actor_num_gpus_per_node {8 if n_actor_gpus > 8 else n_actor_gpus}",
        f"--critic_num_nodes {n_critic_gpus // 8 if n_critic_gpus > 8 else 1}",
        f"--critic_num_gpus_per_node {8 if n_critic_gpus > 8 else n_critic_gpus}",
        f"--train_batch_size {n_actor_gpus * actor_per_device_bs // 4}",
        f"--micro_train_batch_size {actor_per_device_bs // 4}",
        f"--critic_micro_train_batch_size {critic_per_device_bs // 4}",
        f"--rollout_batch_size {n_actor_gpus * actor_per_device_bs}",
        f"--micro_rollout_batch_size {actor_per_device_bs}",
        f"--max_epochs 1",
        f"--prompt_max_len 128",
        f"--generate_max_len {args.seqlen}",
        f"--zero_stage {args.zero_stage}",
        "--normalize_reward",
        "--actor_init_on_gpu",
        "--flash_attn",
        "--gradient_checkpointing",
    ]
    if colocate_actor_critic:
        flags.append("--colocate_actor_critic")
    if colocate_ref_reward:
        flags.append("--colocate_ref_reward")
    if args.offload:
        flags.append("--adam_offload")
    model_path = get_path_from_model_size(args.model_size)
    flags.extend(
        [
            f"--pretrain {model_path}",
            f"--reward_pretrain /lustre/public/pretrained_model_weights/Llama-2-7b-hf",
        ]
    )
    # if args.zero_stage == 3:
    flags.append("--bf16")
    if n_vllm_engines_gpus > 0:
        if args.model_size <= 34:
            vllm_tp_size = 2
        elif args.model_size == 70:
            vllm_tp_size = 4
        assert n_vllm_engines_gpus % vllm_tp_size == 0
        n_vllm_engines = n_vllm_engines_gpus // vllm_tp_size
        flags.append(f"--vllm_num_engines {n_vllm_engines}")
        flags.append(f"--vllm_tensor_parallel_size {vllm_tp_size}")
    return " ".join(cmd + flags)


def control_cmd(expr_name, trial_name, local_mode):
    cmd = f"python3 remote.py controller -e {expr_name} -f {trial_name}"
    if local_mode:
        cmd += " --local_mode"
    return cmd


def ray_cluster_cmd(expr_name, trial_name):
    flags = [f"-e {expr_name}", f"-f {trial_name}"]
    return (
        f"python3 remote.py ray -i {{jobstep_id}} -g {{n_jobsteps}} -r {{worker_submission_index}} "
        f"-p {{wprocs_per_jobstep}} -j {{wprocs_in_job}} -o {{wproc_offset}} {' '.join(flags)}"
    )


def main_ray_driver(args):
    # ray_port = args.ray_port
    # launch ray cluster head
    ray_flags = [
        # f"--port={ray_port}",
        "--head",
        f"--temp-dir={os.path.join('/tmp/ray/', args.experiment_name, args.trial_name)}",
    ]
    # if args.mode == "slurm":
    #     ray_flags.extend(
    #         [
    #             f"--num-cpus=0",
    #             f"--num-gpus=0",
    #         ]
    #     )
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

    jobsub_addr = None
    for l in output.split("\n"):
        if "-- python my_script.py" in l:
            jobsub_addr = l.split("'")[1]
            logger.info("Found Ray job submission address: '%s'", jobsub_addr)
            break
    if jobsub_addr is None:
        raise RuntimeError(f"Address not found in ray start output: {output}.")

    # For slurm model, launch the Ray cluster via slurm command.
    ngpus, _, _, colocate_actor_critic, colocate_ref_reward = get_ngpus_and_nodelist_from_model_size(
        args.model_size,
        scale_actor=args.scale_actor,
        scale_critic=args.scale_critic,
    )
    assert ngpus % 8 == 0

    controller = RayController(
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
        local_mode=args.mode == "local",
        ray_cluster_count=None if args.mode == "local" else (ngpus // 8 - 1),
        job_submission_addr=jobsub_addr,
        driver_cmd=driver_cmd(args),
    )
    try:
        controller.start()
    except (KeyboardInterrupt, Exception) as e:
        subprocess.check_output(f"ray stop", shell=True)
        raise e
    subprocess.check_output(f"ray stop", shell=True)


def allocate_slurm_resource_for_ray_cluster(ntasks, nodelist):
    from scheduler.slurm.utils import (
        get_all_node_resources,
        available_hostnames,
        SlurmResource,
        SlurmResourceNotEnoughException,
    )

    resource_requirement = SlurmResource(
        mem=800000,
        cpu=64,
        gpu_type="tesla",
        gpu=8,
    )
    all_resources = get_all_node_resources()
    valid_hostnames = available_hostnames(nodelist=nodelist)
    valid_hostnames = list(filter(lambda x: x in all_resources, valid_hostnames))
    valid_resources = {hn: all_resources[hn] for hn in valid_hostnames}
    valid_resources = sorted(valid_resources.items(), key=lambda x: x[1], reverse=True)
    task_left = ntasks
    allocated = dict()
    for hostname, resource in valid_resources:
        tmp = task_left
        while task_left > 0:
            try:
                resource = resource - resource_requirement
            except ValueError:
                break
            if not resource.valid():
                break
            task_left -= 1
        if tmp - task_left > 0:
            allocated[hostname] = tmp - task_left
        all_resources[hostname] = resource
    if task_left > 0:
        logger.info(f"Request node list {nodelist} is not empty. Experiment could not run.")
        raise SlurmResourceNotEnoughException()
    hostlist = []
    for hostname, task_num in allocated.items():
        hostlist += [hostname] * task_num
    return "\n".join(hostlist)


def main_start(args):
    image_name = "llm/llm-openrlhf"
    # args.ray_port = ray_port = get_random_port()

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

    ngpus, _, nodelist, colocate_actor_critic, colocate_ref_reward = get_ngpus_and_nodelist_from_model_size(
        args.model_size,
        scale_actor=args.scale_actor,
        scale_critic=args.scale_critic,
    )

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
                f"--global_bs {args.global_bs}",
                f"--zero_stage {args.zero_stage}",
                f"--offload" if args.offload else "",
                f"--seqlen {args.seqlen}",
                "--scale_actor" if args.scale_actor else "",
                "--scale_critic" if args.scale_critic else "",
                # f"--ray_port {ray_port}",
            ]
        )

        assert ngpus % 8 == 0
        n_ray_cluster_nodes = ngpus // 8 - 1

        _hostfile_content = allocate_slurm_resource_for_ray_cluster(ngpus // 8, nodelist)

        head_nodelist = _hostfile_content.split("\n")[0]
        ray_cluster_hostfile_content = "\n".join(_hostfile_content.split("\n")[1:])

        sched.submit_array(
            worker_type="ctl",
            cmd=remote_driver_cmd,
            count=1,
            cpu=64,
            gpu=8,
            gpu_type="tesla",
            mem=int(1000e3),
            env_vars=base_environs,
            container_image=image_name,
            time_limit=CONTROLLER_TIME_LIMIT,
            nodelist=head_nodelist,
        )

        logger.debug(f"Scheduling Ray cluster...")

        raycluster_slurm_job_name = None
        if n_ray_cluster_nodes > 0:
            job_environs = {**base_environs, **_LLM_ENVVARS}
            ray_cluster_log = os.path.join(LOG_ROOT, expr_name, trial_name, "ray_cluster.log")
            os.makedirs(os.path.dirname(ray_cluster_log), exist_ok=True, mode=0o775)
            hostfile_path = os.path.join(LOG_ROOT, expr_name, trial_name, "ray_cluster.hostfile")
            multiprog_path = os.path.join(LOG_ROOT, expr_name, trial_name, "ray_cluster.multiprog")
            logger.info(f"To check the output, run \n\t`tail -f {ray_cluster_log}`.")

            multiprog_cmd = ray_cluster_cmd(args.experiment_name, args.trial_name).format(
                jobstep_id="%t",
                n_jobsteps=n_ray_cluster_nodes,
                worker_submission_index=0,
                wprocs_per_jobstep=1,
                wprocs_in_job=n_ray_cluster_nodes,
                wproc_offset=0,
            )
            multiprog_content = f"0-{n_ray_cluster_nodes - 1} {multiprog_cmd}\n"
            with open(multiprog_path, "w") as f:
                f.write(multiprog_content)

            with open(hostfile_path, "w") as f:
                f.write(ray_cluster_hostfile_content)

            raycluster_slurm_job_name = f"{args.experiment_name}_{args.trial_name}:ray_cluster"
            sbatch_lines = [
                "#!/bin/bash",
                f"#SBATCH --job-name={raycluster_slurm_job_name}",
                f"#SBATCH --output={ray_cluster_log}",
                f"#SBATCH --ntasks={n_ray_cluster_nodes}",
                f"#SBATCH --ntasks-per-node=1",
                f"#SBATCH --gpus-per-task=tesla:8",
                f"#SBATCH --cpus-per-task=64",
                f"#SBATCH --mem-per-cpu={1000000 // 64}M",
                "#SBATCH --distribution=arbitrary",
            ]

            srun_env = os.environ.copy()
            srun_env["SLURM_HOSTFILE"] = hostfile_path
            # Setup step command.
            srun_flags = [
                f"--ntasks={n_ray_cluster_nodes}",
                f"--cpus-per-task=64",
                f"--gpus-per-task=tesla:8",
                f"--mem-per-cpu={1000000 // 64}",
                f"--export={','.join(str(k)+'='+str(v) for k, v in job_environs.items())}",
                f"--multi-prog",
                f"--container-image={image_name}",
                f"--container-mounts={cluster.spec.default_mount}",
                f"--container-mount-home",
                "--container-workdir=/home/fw/sosp-workspace/OpenRLHF",
            ]

            srun_cmd = f'srun -l {" ".join(srun_flags)} {multiprog_path}'

            sbatch_lines += [
                'echo "[Runner] StartTime: $(date -u)"',
                'echo "[Runner] Host: $(hostname)"',
                "echo '[Runner] Command: {}'".format(srun_cmd),
                "echo '[Runner] Log: {}'".format(ray_cluster_log),
                'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
                'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
                srun_cmd,
                "RETCODE=$?",
                'echo "[Runner] FinishTime: $(date -u)"',
                'echo "[Runner] RetCode: $RETCODE"',
                'echo "[Runner] ------------"',
                "exit $RETCODE",
            ]

            script_strs = "\n".join(list(filter(lambda x: x, sbatch_lines))) + "\n"
            script = script_strs.encode("ascii")

            r = (
                subprocess.check_output(["sbatch", "--parsable"], input=script, env=srun_env)
                .decode("ascii")
                .strip()
            )
        from scheduler.slurm.utils import cancel_jobs

        new_logfile = os.path.join(LOG_ROOT, args.experiment_name, args.trial_name, f"rlhf.log")
        logger.info(f">>>>> Check the Ray experiment log with: <<<<<\n\ttail -f {new_logfile} ")
        try:
            sched.wait()
        except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
            if raycluster_slurm_job_name is not None:
                cancel_jobs([raycluster_slurm_job_name])
            sched.stop_all()
            raise e
        if raycluster_slurm_job_name is not None:
            cancel_jobs([raycluster_slurm_job_name])


def get_path_from_model_size(model_size: int):
    if model_size == 7:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
    elif model_size == 13:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
    elif model_size == 34:
        model_path = "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf-pt"
    elif model_size == 70:
        model_path = "/lustre/public/pretrained_model_weights/Llama-2-70b-hf"
    else:
        raise NotImplementedError()
    return model_path


def get_ngpus_and_nodelist_from_model_size(
    model_size: int,
    scale_actor: bool,
    scale_critic: bool,
):
    assert scale_critic or scale_actor
    colocate_actor_critic = False
    colocate_ref_reward = False
    if model_size in [7]:
        # actor, critic, ref, rew, vllm-engine
        if scale_actor and not scale_critic:
            device_partition = (4, 2, 1, 1, 0)
        else:
            colocate_actor_critic = True
            device_partition = (4, 4, 1, 1, 2)
        ngpus, nodelist = 8, "QH-com22"
    elif model_size == 13:
        if scale_actor and not scale_critic:
            device_partition = (8, 4, 2, 2, 0)
        else:
            colocate_actor_critic = True
            device_partition = (8, 8, 2, 2, 4)
        ngpus, nodelist = 16, "QH-com[22,24]"
    elif model_size in [34]:
        if scale_actor and not scale_critic:
            device_partition = (16, 4, 4, 2, 6)
        else:
            colocate_actor_critic = True
            device_partition = (16, 16, 4, 8, 4)
        ngpus, nodelist = 32, "QH-com[41,46-48]"
    elif model_size == 70:
        if scale_actor and not scale_critic:
            device_partition = (32, 8, 8, 4, 12)
        else:
            colocate_actor_critic = True
            device_partition = (48, 48, 4, 8, 4)
        ngpus, nodelist = 64, "QH-com[25,27-29,42-45]"
    if colocate_actor_critic:
        actor_critic_gpus = device_partition[0]
    else:
        actor_critic_gpus = device_partition[0] + device_partition[1]
    if colocate_ref_reward:
        ref_rew_gpus = device_partition[2]
    else:
        ref_rew_gpus = device_partition[2] + device_partition[3]
    assert ngpus == actor_critic_gpus + ref_rew_gpus + device_partition[4]
    return ngpus, device_partition, nodelist, colocate_actor_critic, colocate_ref_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="launch ray cluster and run workers")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--mode", default="slurm", choices=["local", "slurm"])

    subparser.add_argument("--model_size", type=int, choices=[7, 13, 34, 70], required=True)
    subparser.add_argument("--global_bs", type=int, default=4)
    subparser.add_argument("--offload", action="store_true")
    subparser.add_argument("--zero_stage", type=int, default=2)
    subparser.add_argument("--seqlen", type=int, default=256)
    subparser.add_argument("--scale_actor", action="store_true")
    subparser.add_argument("--scale_critic", action="store_true")
    subparser.set_defaults(func=main_start)

    subparser = subparsers.add_parser("driver", help="launch ray cluster and run workers")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--mode", default="slurm", choices=["local", "slurm"])

    subparser.add_argument("--model_size", type=int, choices=[7, 13, 34, 70], required=True)
    subparser.add_argument("--global_bs", type=int, default=4)
    subparser.add_argument("--offload", action="store_true")
    subparser.add_argument("--seqlen", type=int, default=256)
    subparser.add_argument("--zero_stage", type=int, default=2)
    subparser.add_argument("--scale_actor", action="store_true")
    subparser.add_argument("--scale_critic", action="store_true")

    # subparser.add_argument("--ray_port", type=int, required=True)
    subparser.set_defaults(func=main_ray_driver)

    args = parser.parse_args()
    args.func(args)
