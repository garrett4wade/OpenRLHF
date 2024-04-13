import math
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union
import multiprocessing as mp

import ray
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed
import pynvml
import time
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, SwitchBalancingLoss, ValueLoss
from openrlhf.models.utils import masked_mean

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer

import logging

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("DeepSpeed Slurm Launch")
writer = None

def gpu_utilization_monitor(gpu_idx:int, ttl:float):
    pynvml.nvmlInit()
    tik = time.time()
    while time.time() - tik < ttl:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (1024 ** 2)  # Convert bytes to megabytes
        used_memory = memory_info.used / (1024 ** 2)
        memory_usage_percentage = (used_memory / total_memory) * 100
        logger.info(f"GPU {gpu_idx}: Compute Utilization - {utilization.gpu}%, Total Memory - {total_memory:.2f}MB, Used Memory - {used_memory:.2f}MB, Memory Usage - {memory_usage_percentage:.2f}%")
        time.sleep(10)
    pynvml.nvmlShutdown()
    
def get_hf_configs(hf_config):
    num_layers = getattr(hf_config, "num_hidden_layers",
                         getattr(hf_config, "n_layer", None))
    hidden_size = getattr(hf_config, "hidden_size",
                          getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), ("Could not determine number of layers, hidden size, and vocab size of the model", hf_config)

    return num_layers, hidden_size, vocab_size

def calculate_flops(checkpoint_activations_factor, batch_size, seq_length,
                    hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size *
                           seq_length * num_layers * (hidden_size**2)) * (
                               1.0 + (seq_length / (6.0 * hidden_size)) +
                               (vocab_size /
                                (16.0 * num_layers * hidden_size)))
    return flops_per_iteration

def print_throughput_step3(actor_hf_config, critic_hf_config,
                           n_actor_gpus, n_critic_gpus,
                           args,
                           e2e_time,
                           gen_exp_time,
                           train_time,
                           rank=0):
    if rank != 0:
        return
    max_prompt_len = args.prompt_max_len
    max_answer_len = args.generate_max_len

    actor_num_layers, actor_hidden_size, actor_vocab_size = get_hf_configs(
        actor_hf_config)
    critic_num_layers, critic_hidden_size, critic_vocab_size = get_hf_configs(
        critic_hf_config)

    seq_length = max_prompt_len + max_answer_len
    batch_size = args.rollout_batch_size
    samples_per_second = batch_size / e2e_time

    actor_checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
    critic_checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3

    actor_num_params = actor_hf_config._num_params
    actor_params_in_billions = actor_num_params / (1e9)

    critic_num_params = critic_hf_config._num_params
    critic_params_in_billions = critic_num_params / (1e9)

    # Megatron paper's formula to calculate training flops

    actor_train_flops_per_iteration = calculate_flops(
        actor_checkpoint_activations_factor, batch_size, seq_length,
        actor_hf_config)
    critic_train_flops_per_iteration = calculate_flops(
        critic_checkpoint_activations_factor, batch_size, seq_length,
        critic_hf_config)

    total_train_flops = actor_train_flops_per_iteration + critic_train_flops_per_iteration
    actor_train_tflops = actor_train_flops_per_iteration / (train_time * n_actor_gpus *
                                        (10**12))
    critic_train_tflops = critic_train_flops_per_iteration/ (train_time * n_critic_gpus *
                                        (10**12))
    train_tflops = actor_train_tflops + critic_train_tflops

    gen_bs = args.rollout_batch_size

    # Modified formula for calculating flops in the forward pass only
    gen_flops_per_iteration = (
        24 * gen_bs * seq_length * actor_num_layers *
        (actor_hidden_size**2)) * (
            1.0 + (seq_length / (6.0 * actor_hidden_size)) +
            (actor_vocab_size /
                (16.0 * actor_num_layers * actor_hidden_size)))

    gen_tflops = gen_flops_per_iteration / (gen_exp_time * n_actor_gpus *
                                            (10**12))

    if actor_hf_config.torch_dtype == torch.float16:
        num_bytes = 2
    elif actor_hf_config.torch_dtype == torch.float32:
        num_bytes = 4
    else:
        num_bytes = -1

    pertok_lat = gen_exp_time / max_answer_len
    gen_bw = 1 / pertok_lat * actor_num_params * num_bytes / 1e9

    generation_batches = (gen_bs // n_actor_gpus // args.micro_rollout_batch_size)
    total_flops_per_iteration = total_train_flops + gen_flops_per_iteration * generation_batches
    total_tflops = total_flops_per_iteration / (e2e_time * (n_actor_gpus + n_critic_gpus) *
                                                (10**12))

    print(
        f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
    )
    print(
        f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat*1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: {gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {max_answer_len}"
    )
    print(
        f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}"
    )
    actor_param_string = f"{actor_params_in_billions:.3f} B" if actor_params_in_billions != 0 else "NA"
    critic_param_string = f"{critic_params_in_billions:.3f} B" if critic_params_in_billions != 0 else "NA"
    print(
        f"Actor Model Parameters => {actor_param_string}, Critic Model Parameters => {critic_param_string}"
    )

def _count_params(model):
    return sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in model.parameters()
        ])
class PPOTrainer(ABC):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (nn.Module): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(
            actor, critic, reward_model, initial_model, tokenizer, prompt_max_len, self.kl_ctl, strategy, reward_fn
        )
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        self.gpu_monior_proc = None

    def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        args,
        hf_actor_config=None,
        hf_critic_config=None,
    ) -> None:
        # this function will be called in Ray PPO actors
        # so torch.distributed.get_world_size() will return the number of actors
        if hasattr(args, "actor_num_nodes"):
            n_actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
            n_critic_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
        else:
            hf_actor_config = self.experience_maker.actor.model.module.config
            hf_actor_config._num_params = _count_params(self.experience_maker.actor.model.module)
            hf_critic_config = self.experience_maker.critic.module._hf_config
            hf_critic_config._num_params = _count_params(self.experience_maker.critic)
            n_actor_gpus = n_critic_gpus = torch.distributed.get_world_size()
            # launch monitor process for non-ray training
            if self.gpu_monior_proc is None:
                self.gpu_monior_proc = mp.Process(target=gpu_utilization_monitor, args=(torch.cuda.current_device(), 3600))
                self.gpu_monior_proc.start()
        
        _step_cnt = 0
        last_gen_time = last_inf_time = 0
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        global_step = 1

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = prompts_dataloader.__len__() // update_timesteps  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        for episode in range(args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            # pbar = tqdm(
            #     range(self.prompts_dataloader.__len__()),
            #     desc=f"Episode [{episode + 1}/{args.num_episodes}]",
            #     disable=not self.strategy.is_rank_0(),
            # )

            train_iter_tik = time.perf_counter()
            for rand_prompts in self.prompts_dataloader:
                experience, gen_time, inf_time = self.experience_maker.make_experience(rand_prompts, **self.generate_kwargs)
                logger.info(f"Generation sequence length: {experience.sequences.size(1)}")
                # print prompt/answer in each update step
                # if global_step % update_timesteps == 0:
                #     output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
                #     self.strategy.print(output[0])
                last_gen_time += gen_time
                last_inf_time += inf_time
                self.replay_buffer.append(experience)
                print(">>>> update timesteps", update_timesteps)

                if global_step % update_timesteps == 0:
                    torch.cuda.empty_cache()
                    ts = time.perf_counter()
                    self.replay_buffer.normalize("advantages", self.strategy)
                    status = self.ppo_train()
                    _step_cnt += 1
                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()
                    # self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    # logs/checkpoints
                    # self.save_logs_and_checkpoints(args, global_step // update_timesteps, pbar, status)
                    train_time = time.perf_counter() - ts
                    e2e_time = time.perf_counter() - train_iter_tik
                    print_throughput_step3(
                        actor_hf_config=hf_actor_config,
                        critic_hf_config=hf_critic_config,
                        n_actor_gpus=n_actor_gpus,
                        n_critic_gpus=n_critic_gpus,
                        args=args,
                        e2e_time=e2e_time,
                        gen_exp_time=last_gen_time,
                        train_time=train_time,
                        rank=torch.distributed.get_rank(),
                    )
                    last_gen_time = last_inf_time = 0
                    train_iter_tik = time.perf_counter()

                # pbar.update()
                global_step = global_step + 1
                if _step_cnt >= 10:
                    if torch.distributed.get_rank() == 0:
                        print(f">>>>>>>>>>>>>> Benchmarking finishes after {_step_cnt} steps")
                    if self.gpu_monior_proc is not None:
                        self.gpu_monior_proc.kill()
                    return
        if self.gpu_monior_proc is not None:
            self.gpu_monior_proc.kill()

    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        print(f">>>>> train sample batch size {self.replay_buffer.sample_batch_size}, dataloader length {len(dataloader)}")
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                # weighted mean for kl
                status["kl"] *= status["response_length"]
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]

                status_list.append(status)
                short_status = {
                    "pg": status["policy_loss"],
                    "rm": status["reward"],
                    "ret": status["return"],
                    "glen": status["response_length"],
                    "tlen": status["total_length"],
                    "kl": status["kl"],
                }
                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        status = self.training_step_actor(experience)
        status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        num_actions = experience.action_mask.size(1)
        # actor loss
        t1 = time.perf_counter()
        action_log_probs, output = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        )
        fwd_time = time.perf_counter() - t1

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            experience.action_log_probs,
            experience.advantages,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef
        t2 = time.perf_counter()
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {
            "policy_loss": actor_loss.item(),
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        bwd_time = time.perf_counter() - t2
        print(f">>>> Actor forward time {fwd_time:.2f}s, backward time {bwd_time:.2f}s, #seqs={experience.attention_mask.shape[0]}")
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # critic loss
        t1 = time.perf_counter()
        values, output = self.critic(
            experience.sequences,
            action_mask=experience.action_mask,
            attention_mask=experience.attention_mask,
            return_output=True,
        )
        fwd_t = time.perf_counter() - t1
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            experience.values,
            experience.returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        t2 = time.perf_counter()
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
        }
        bwd_t = time.perf_counter() - t2
        print(f">>>> critic forward time {fwd_t:.2f}s, backward time {bwd_t:.2f}s, #seqs={experience.attention_mask.shape[0]}")
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            step_bar.set_postfix(logs_dict)
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.actor.model, os.path.join(args.ckpt_path, "_actor"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
            self.strategy.save_ckpt(
                self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
