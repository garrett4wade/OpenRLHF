import inspect
import logging
import multiprocessing as mp
import os
import time
from functools import partial
from typing import *

import pynvml
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("DeepSpeed Slurm Launch")


def gpu_utilization_monitor(name, gpu_idx: int, ttl: float):
    pynvml.nvmlInit()
    tik = time.time()
    while time.time() - tik < ttl:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (1024**2)  # Convert bytes to megabytes
        used_memory = memory_info.used / (1024**2)
        memory_usage_percentage = (used_memory / total_memory) * 100
        logger.info(
            f"Model {name} GPU {gpu_idx}: Compute Utilization - {utilization.gpu}%, Total Memory - {total_memory:.2f}MB, Used Memory - {used_memory:.2f}MB, Memory Usage - {memory_usage_percentage:.2f}%"
        )
        time.sleep(10)
    pynvml.nvmlShutdown()


@ray.remote
class LLMRayActor:
    def __init__(self, idx, *args, **kwargs):
        self.llm = LLM(*args, **kwargs)

        if idx == 0:
            self._gpu_monitor_proc = mp.Process(target=gpu_utilization_monitor, args=(f"vLLM", idx, 7200))
            self._gpu_monitor_proc.start()

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name):
        return self.llm.llm_engine._run_workers(
            "init_process_group", master_address, master_port, rank_offset, world_size, group_name
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.llm_engine._run_workers("update_weight", name, dtype, shape, empty_cache)


if __name__ == "__main__":
    # ray.init(num_gpus=4, num_cpus=4)
    tp_size = 4
    bundles = [{"GPU": 1, "CPU": 1}] * tp_size
    pg = placement_group(bundles)
    ray.get(pg.ready())

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )
    llm = LLMRayActor.options(
        num_cpus=1,
        num_gpus=0.1,
        scheduling_strategy=scheduling_strategy,
    ).remote(0, "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf-pt", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
