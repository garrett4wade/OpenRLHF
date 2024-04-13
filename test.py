
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import vllm
import torch
import importlib
import os
from vllm import LLM
from vllm.model_executor.weight_utils import hf_model_weights_iterator
from vllm.worker.worker import Worker

import pynvml
import time
import multiprocessing as mp

import openrlhf
import openrlhf.trainer.ray.vllm_engine
import openrlhf.trainer
import openrlhf.trainer.ray
from openrlhf.utils.distributed_util import init_process_group

import deepspeed
import transformers.deepspeed
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):

        self.llm = LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


if __name__ == "__main__":
    llama7b_ckpt_path = "/lustre/public/pretrained_model_weights/Llama-2-7b-hf"

    ray.init(num_gpus=4, num_cpus=4)
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
        num_gpus=0,
        scheduling_strategy=scheduling_strategy,
    ).remote(llama7b_ckpt_path, tensor_parallel_size=tp_size)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
