
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import sys
import vllm
import torch
import importlib
import os
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
from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)



if __name__ == "__main__":
    llama7b_ckpt_path = "/lustre/public/pretrained_model_weights/llama2/Llama-2-7b-hf-config-only/"

    ray.init()

    engines = create_vllm_engines(
        num_engines=2,
        tensor_parallel_size=2,
        pretrain=llama7b_ckpt_path,
        seed=0,
        enable_prefix_caching=False,
        max_model_len=2048,
    )
    # tp_size = 4
    # bundles = [{"GPU": 1, "CPU": 1}] * tp_size
    # pg = placement_group(bundles)
    # ray.get(pg.ready())

    # scheduling_strategy = PlacementGroupSchedulingStrategy(
    #     placement_group=pg,
    #     placement_group_capture_child_tasks=True,
    #     placement_group_bundle_index=0,
    # )
    # llm = LLMRayActor.options(
    #     num_cpus=1,
    #     num_gpus=0,
    #     scheduling_strategy=scheduling_strategy,
    # ).remote(llama7b_ckpt_path, tensor_parallel_size=tp_size)
    output = ray.get([engine.generate.remote("San Franciso is a") for engine in engines])
    print(f"output: {output}")
