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

# LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
# DATE_FORMAT = "%Y%m%d-%H:%M:%S"
# logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
# logger = logging.getLogger("DeepSpeed Slurm Launch")


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

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
class LLMRayActor:
    def __init__(self, idx, *args, **kwargs):
        import vllm
        print(">>>>>>>>>>>>>>>>", os.getenv("CUDA_VISIBLE_DEVICES"))
        # print(">>>>>>>>>>>>>>>>", os.getenv("CUDA_VISIBLE_DEVICES"))

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.4.1":
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = "openrlhf.trainer.ray.vllm_worker_wrap"
                    kwargs["worker_class_name"] = "WorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)

        # if idx == 0:
        #     self._gpu_monitor_proc = mp.Process(target=gpu_utilization_monitor, args=(f"vLLM", idx, 7200))
        #     self._gpu_monitor_proc.start()

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
):
    vllm_engines = []
    for i in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if tensor_parallel_size > 1:
            num_gpus = 0.1
            bundles = [{"GPU": 1, "CPU": 2}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                i,
                pretrain,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="float16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
                disable_custom_all_reduce=True,
                load_format="dummy",
            )
        )

    return vllm_engines


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
