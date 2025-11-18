# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
import json
from unittest.mock import patch

import deepspeed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather_object
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from .utils import pad, pad_with_batch
from collections import Counter
import random
import copy
import statistics
import scipy.stats as stats

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from vllm import LLM, SamplingParams

# AutoModelForCausalLM.register(config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLAdaptGRPOVLLMTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        group_prefix: Optional[str] = None,
        use_cache_generation: bool = False,
        tasks: Optional[dict[str, Any]] = None,
        freeze_vis_encoder: Optional[bool] = False,
        freeze_backbone: Optional[bool] = False,
        dynamic_mode_strategy: Optional[str] = "disable",
        mask_prefix_for_loss: Optional[bool] = False,
        generation_temperature: Optional[float] = 0.9,
        average_by_token: Optional[bool] = False,
        adaptive_strategy: Optional[str] = "in_batch",
        advantage_strategy: Optional[str] = "default",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        self.use_cache_generation = use_cache_generation
        self.mask_prefix_for_loss = mask_prefix_for_loss
        self.average_by_token = average_by_token
        self.adaptive_strategy = adaptive_strategy
        self.advantage_strategy = advantage_strategy
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            print("current model init kwargs", model_init_kwargs)
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            if "torch_dtype" not in model_init_kwargs:
                model_init_kwargs["torch_dtype"] = torch.bfloat16
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)

            elif "Qwen2.5-VL" in model_id or 'niesheng' in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)

            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled() or peft_config is None:
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id or 'niesheng' in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            # self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or 'niesheng' in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        else:
            processing_class = processing_class.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.group_prefix = group_prefix
        self.group_num = len(group_prefix) if group_prefix is not None else 1
        # make sure the number of generations is divisible by the group size
        assert args.num_generations % self.group_num == 0
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.dynamic_mode_strategy = dynamic_mode_strategy
        self.dynamic_finished_flag = False # intialize with False (the dynamic process has not finished)
        if self.adaptive_strategy == "in_batch":
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,  
                temperature=generation_temperature, # HACK
                top_p=0.9,
                top_k=50,
                num_return_sequences=self.num_generations//self.group_num,
                pad_token_id=pad_token_id,
            )
        else:
            raise NotImplementedError
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.tasks = tasks if tasks is not None else ['all']
        self.task_metrics = {}
        for task in self.tasks:
            self.think_mode = ["mode_{}".format(i) for i in range(len(self.group_prefix))] if self.group_prefix is not None else ["mode"]
            self.task_metrics[task] = {m: 0.0 for m in self.think_mode}

        if freeze_vis_encoder:
            model.visual.requires_grad_(False)
        if freeze_backbone:
            model.model.requires_grad_(False)
        if freeze_vis_encoder and freeze_backbone:
            # if all parameters are frozen, activate the projector (only used during debug for saving GPU memory)
            model.visual.merger.requires_grad_(True)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # initialize VLLM
        if self.accelerator.is_main_process:
            # load vllm
            vllm_device = "auto"
            if vllm_device == "auto":
                vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
            # Check that the requested device is available
            if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(
                    f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                    "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                    "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                    f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                )
            # Check that the requested device is not also used for training
            if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                print(
                    f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                    "behavior. It is recommended to use a dedicated device for vLLM."
                )
            # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
            # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
            # setting (profiling_patch).
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
            )
            with world_size_patch, profiling_patch:
                self.llm = LLM(
                    model=model.name_or_path,
                    device=vllm_device,
                    gpu_memory_utilization=0.7,
                    # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                    # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                    # This is particularly useful here because we generate completions from the same prompts.
                    enable_prefix_caching=True,
                    distributed_executor_backend="external_launcher", seed=42
                )
                self.sampling_params = SamplingParams(
                    temperature=generation_temperature,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=self.max_completion_length,
                )

        self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs       
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert len(inputs) == 1, "Current method only supports batch size=1"
        device = self.accelerator.device
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        for x in inputs:
            x['prompt'] = json.loads(x['prompt'])
        prompts = [x["prompt"] for x in inputs]
        # print(f'inputs-keys: {inputs[0].keys()}', flush=True)
        # print(f'example-0: {inputs[0]}', flush=True)

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        if self.group_prefix is not None:
            assert len(prompts_text) == 1, "Current method only supports batch size=1"
            grouped_prompts_text = [prompts_text[0] + g for g in self.group_prefix]
            batch_size = self.num_generations//self.group_num
            total_batch_size = self.num_generations
        else:
            batch_size = self.num_generation
            total_batch_size = batch_size
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # the raw sequence information
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]
        prompt_length = prompt_ids.shape[1]

        
        if self.max_prompt_length is not None:
            # TODO: maybe bug here, since truncated prompt_ids will be used in the generation
            pass
            # prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            # prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                # remove_hooks(model)
                unwrapped_model = self.accelerator.unwrap_model(model)
                if is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step
        
        # Prepare inputs for vLLM generation
        inputs_vllm = []
        input_ids = []
        group_prefix_mask = []
        if self.group_prefix is not None:
            # generate the completions for each group with different prefix
            group_completion_ids = []
            group_prefix_mask = []
            # go through all modes
            grouped_prompts_text_to_generate = grouped_prompts_text
            for g in grouped_prompts_text_to_generate:
                prompt_inputs_group = self.processing_class(
                    text=[g],
                    input_image_grid_thw=image_grid_thw,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                prompt_inputs_group = super()._prepare_inputs(prompt_inputs_group)
                for i in range(batch_size):
                    inputs_vllm.append(
                        {
                            "prompt": g,
                            "multi_modal_data": {
                                "image": images
                            },
                        }
                    )
                    input_ids.append(prompt_inputs_group['input_ids'])
        else:
            for i in range(batch_size):
                inputs_vllm.append(
                    {
                        "prompt": prompts_text[0],
                        "multi_modal_data": {
                            "image": images
                        },
                    }
                )
                input_ids.append(prompt_inputs['input_ids'])

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_inputs_vllm = gather_object(inputs_vllm)
        if self.accelerator.is_main_process:
            outputs = self.llm.generate(all_inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)
            completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
        else:
            completion_ids = [None] * len(all_inputs_vllm)

        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice.
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        process_slice = slice(
            self.accelerator.process_index * len(prompts) * total_batch_size,
            (self.accelerator.process_index + 1) * len(prompts) * total_batch_size,
        )
        completion_ids = completion_ids[process_slice]

        # Pad the completions, and concatenate them with the prompts
        prompt_completion_ids = []
        group_prefix_mask = []
        for cur_input_ids, ids in zip(input_ids,completion_ids):
            cur_prompt_completion_ids = torch.cat([cur_input_ids[0], torch.tensor(ids, device=device)])
            prompt_completion_ids.append(cur_prompt_completion_ids)
            # get the prefix_mask
            current_prefix_mask = torch.zeros_like(cur_prompt_completion_ids)
            current_prefix_mask[:cur_input_ids.shape[1]] = 1
            group_prefix_mask.append(current_prefix_mask)

        prompt_completion_ids = pad(prompt_completion_ids, padding_value=self.processing_class.pad_token_id)
        group_prefix_mask = pad(group_prefix_mask, padding_value=0)[:, prompt_length:]
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = prompt_inputs["pixel_values"].repeat_interleave(self.num_generations, dim=0).view(-1, pixel_values.shape[-1])
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}

                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # print(f'keys: {inputs[0].keys()}')
                # print(f'prompts:{prompts}',flush=True)
                # print(f'completions: {completions}',flush=True)
                # print(f'reward_kwargs: {reward_kwargs}',flush=True)

                reward_kwargs['image_grid_thw'] = image_grid_thw
                reward_kwargs['max_steps'] = self.state.max_steps
                reward_kwargs['global_step'] = self.state.global_step
                reward_kwargs['accuracy_reward'] = rewards_per_func[:, 0].detach().tolist()
                reward_kwargs['group_prefix'] = self.group_prefix
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if self.advantage_strategy == "default":
            # Sum the rewards from all reward functions
            rewards = rewards_per_func.sum(dim=1) #* original implement: acc-reward vs format-reward=1:1

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        elif self.advantage_strategy == "v2":
            # split the rewards into multiple groups
            rewards_per_group = torch.split(rewards_per_func, self.num_generations//self.group_num, dim=0)
            rewards = rewards_per_func.sum(dim=1)

            # compute the advantages within each group
            in_group_advantages = []
            for in_group_reward in rewards_per_group:
                # compute the mean and std of the rewards in each group
                in_group_reward_sumed = in_group_reward.sum(dim=1)
                mean_grouped_rewards = in_group_reward_sumed.view(-1, self.num_generations//self.group_num).mean(dim=1)
                std_grouped_rewards = in_group_reward_sumed.view(-1, self.num_generations//self.group_num).std(dim=1)

                # Normalize the rewards to compute the advantages
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations//self.group_num, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations//self.group_num, dim=0)
                in_group_advantages.append((in_group_reward_sumed - mean_grouped_rewards) / (std_grouped_rewards + 1e-4))
            in_group_advantages = torch.cat(in_group_advantages)
            
            # compute the group_preference / group-level rewards (mainly based on accuracy reward now)
            group_avg_rewards = rewards_per_func[:, 0].reshape(self.group_num, -1).mean(dim=1)
            group_avg_rewards_mean = group_avg_rewards.view(-1, self.group_num).mean(dim=1)
            group_avg_rewards_std = group_avg_rewards.view(-1, self.group_num).std(dim=1)

            # normalize group_preference
            group_avg_rewards_mean = group_avg_rewards_mean.repeat_interleave(self.group_num, dim=0)
            group_avg_rewards_std = group_avg_rewards_std.repeat_interleave(self.group_num, dim=0)
            group_relative_advantages = (group_avg_rewards - group_avg_rewards_mean)
            group_relative_advantages = group_relative_advantages.repeat_interleave(self.num_generations//self.group_num, dim=0)
            
            # in this strategy, the adavantages should be with the same shape as the per_token_logps
            # [batch_size, seq_len]
            advantages = (1-group_prefix_mask)*in_group_advantages.unsqueeze(1) + group_prefix_mask*group_relative_advantages.unsqueeze(1)
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        elif self.advantage_strategy == "v3":
            # split the rewards into multiple groups
            rewards = rewards_per_func.sum(dim=1)

            # compute the rollout advantages across all trajectories in different groups
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            rollout_advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
            # compute the group_preference / group-level rewards (mainly based on accuracy reward now)
            group_avg_rewards = rewards_per_func[:, 0].reshape(self.group_num, -1).mean(dim=1)
            group_avg_rewards_mean = group_avg_rewards.view(-1, self.group_num).mean(dim=1)
            group_avg_rewards_std = group_avg_rewards.view(-1, self.group_num).std(dim=1)

            # normalize group_preference
            group_avg_rewards_mean = group_avg_rewards_mean.repeat_interleave(self.group_num, dim=0)
            group_avg_rewards_std = group_avg_rewards_std.repeat_interleave(self.group_num, dim=0)
            group_relative_advantages = (group_avg_rewards - group_avg_rewards_mean)
            group_relative_advantages = group_relative_advantages.repeat_interleave(self.num_generations//self.group_num, dim=0)
            
            # in this strategy, the adavantages should be with the same shape as the per_token_logps
            # [batch_size, seq_len]
            advantages = (1-group_prefix_mask)*rollout_advantages.unsqueeze(1) + group_prefix_mask*group_relative_advantages.unsqueeze(1)
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        elif self.advantage_strategy == "v3-2":
            # split the rewards into multiple groups
            rewards = rewards_per_func.sum(dim=1)

            # compute the rollout advantages across all trajectories in different groups
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            rollout_advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
            # compute the group_preference / group-level rewards (mainly based on accuracy reward now)
            group_avg_rewards = rewards_per_func[:, 0].reshape(self.group_num, -1).mean(dim=1)
            group_avg_rewards_mean = group_avg_rewards.view(-1, self.group_num).mean(dim=1)
            group_avg_rewards_std = group_avg_rewards.view(-1, self.group_num).std(dim=1)

            # normalize group_preference
            group_avg_rewards_mean = group_avg_rewards_mean.repeat_interleave(self.group_num, dim=0)
            group_avg_rewards_std = group_avg_rewards_std.repeat_interleave(self.group_num, dim=0)
            group_relative_advantages = (group_avg_rewards - group_avg_rewards_mean)*2 # scale of 2 to encourage group preference
            group_relative_advantages = group_relative_advantages.repeat_interleave(self.num_generations//self.group_num, dim=0)
            
            # in this strategy, the adavantages should be with the same shape as the per_token_logps
            # [batch_size, seq_len]
            advantages = (1-group_prefix_mask)*rollout_advantages.unsqueeze(1) + group_prefix_mask*group_relative_advantages.unsqueeze(1)
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        elif self.advantage_strategy == "v3-3":
            # split the rewards into multiple groups
            rewards = rewards_per_func.sum(dim=1)

            # compute the rollout advantages across all trajectories in different groups
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            rollout_advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
            # compute the group_preference / group-level rewards (mainly based on accuracy reward now)
            group_avg_rewards = rewards_per_func[:, 0]
            group_avg_rewards_normed = (group_avg_rewards - group_avg_rewards.mean().unsqueeze(0)) / (group_avg_rewards.std().unsqueeze(0)+1e-4)

            # normalize group_preference
            group_relative_advantages = group_avg_rewards_normed.reshape(self.group_num, -1).mean(dim=1)
            group_relative_advantages = group_relative_advantages.repeat_interleave(self.num_generations//self.group_num, dim=0)
            
            # in this strategy, the adavantages should be with the same shape as the per_token_logps
            # [batch_size, seq_len]
            advantages = (1-group_prefix_mask)*rollout_advantages.unsqueeze(1) + group_prefix_mask*group_relative_advantages.unsqueeze(1)
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        elif self.advantage_strategy == "v4":
            assert self.group_num == 2, "this function only supports 2 groups"
            # split the rewards into multiple groups
            rewards = rewards_per_func.sum(dim=1)

            # compute the rollout advantages across all trajectories in different groups
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            rollout_advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
            # compute the group_preference / group-level rewards (mainly based on accuracy reward now)
            group_avg_rewards = rewards_per_func[:, 0].detach().tolist()
            def gauss_diff(nums1, nums2):
                mean1 = statistics.mean(nums1)
                mean2 = statistics.mean(nums2)
                var1 = statistics.variance(nums1)
                var2 = statistics.variance(nums2)
                mean_diff = mean1 - mean2
                var_diff = var1 + var2
                return 1 - stats.norm.cdf((0-mean_diff)/(var_diff**0.5 + 1e-5))
            win_prob1 = gauss_diff(group_avg_rewards[:self.num_generations//self.group_num], group_avg_rewards[self.num_generations//self.group_num:])
            group_avg_rewards = torch.tensor([win_prob1, 1-win_prob1], dtype=rollout_advantages.dtype, device=device)
            group_avg_rewards_mean = group_avg_rewards.view(-1, self.group_num).mean(dim=1)

            # normalize group_preference
            group_avg_rewards_mean = group_avg_rewards_mean.repeat_interleave(self.group_num, dim=0)
            group_relative_advantages = (group_avg_rewards - group_avg_rewards_mean)*2 # scale of 2 to encourage group preference
            group_relative_advantages = group_relative_advantages.repeat_interleave(self.num_generations//self.group_num, dim=0)
            
            # in this strategy, the adavantages should be with the same shape as the per_token_logps
            # [batch_size, seq_len]
            advantages = (1-group_prefix_mask)*rollout_advantages.unsqueeze(1) + group_prefix_mask*group_relative_advantages.unsqueeze(1)
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        elif self.advantage_strategy == "adaGRPO":
            assert self.group_num == 2, "this function only supports 2 groups"
            # split the rewards into multiple groups
            rewards = rewards_per_func.sum(dim=1)

            # compute the rollout advantages across all trajectories in different groups
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            rollout_advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
            # compute the group_preference / group-level rewards (mainly based on accuracy reward now)
            group_avg_rewards = rewards_per_func[:, 0].detach().tolist()
            def gauss_diff(nums1, nums2):
                mean1 = statistics.mean(nums1)
                mean2 = statistics.mean(nums2)
                var1 = statistics.variance(nums1)
                var2 = statistics.variance(nums2)
                mean_diff = mean1 - mean2
                var_diff = var1 + var2
                return 1 - stats.norm.cdf((0-mean_diff)/(var_diff**0.5 + 1e-5))
            win_prob1 = gauss_diff(group_avg_rewards[:self.num_generations//self.group_num], group_avg_rewards[self.num_generations//self.group_num:])
            group_avg_rewards = torch.tensor([win_prob1, 1-win_prob1], dtype=rollout_advantages.dtype, device=device)
            group_avg_rewards_mean = group_avg_rewards.view(-1, self.group_num).mean(dim=1)

            # normalize group_preference
            group_avg_rewards_mean = group_avg_rewards_mean.repeat_interleave(self.group_num, dim=0)
            group_relative_advantages = (group_avg_rewards - group_avg_rewards_mean)
            group_relative_advantages = group_relative_advantages.repeat_interleave(self.num_generations//self.group_num, dim=0)
            
            # in this strategy, the adavantages should be with the same shape as the per_token_logps
            # [batch_size, seq_len]
            advantages = (1-group_prefix_mask)*rollout_advantages.unsqueeze(1) + group_prefix_mask*group_relative_advantages.unsqueeze(1)
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        else:
            raise NotImplementedError

        if self.beta == 0: #! control kl optimization
            # print(f'Not optimizing KL loss here...' ,flush=True) 
            per_token_loss = -(per_token_loss) # not optimize kl loss 
        else:
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        if self.average_by_token:
            loss = torch.masked_select(per_token_loss, completion_mask.bool()).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        over_long_seqs = self.accelerator.gather_for_metrics(completion_mask.sum(1)>=self.max_completion_length).float().mean().item()
        self._metrics["over_lengthy_sequences"].append(over_long_seqs)

        # adaptive metrics
        grounded_prop, format_confidence = self.adaptive_metrics(completions)
        gathred_grounded_prop = self.accelerator.gather_for_metrics([grounded_prop])
        gathred_format_confidence = self.accelerator.gather_for_metrics([format_confidence])
        self._metrics["grounded_proportion"].append(sum(gathred_grounded_prop)/len(gathred_grounded_prop))
        self._metrics["format_confidence"].append(sum(gathred_format_confidence)/len(gathred_format_confidence))

        # gathere important information across processes
        gathered_reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func)
        response_modes = self.completion_to_mode(completions)
        gathered_response_modes = self.accelerator.gather_for_metrics(response_modes) # n_processes * n_generations
        if self.tasks[0] == 'all':
            # If the task is 'all', we gather the rewards for all tasks
            gathered_task = ['all'] * self.accelerator.num_processes
        else:
            # If the task is not 'all', we gather the rewards for each task
            # Note: this assumes that the tasks are provided in the same order across all processes
            gathered_task = self.accelerator.gather_for_metrics([inputs[0]["question_format"]]*len(completions)) if "question_format" in inputs[0] else ['all'] * self.accelerator.num_processes
        try:
            self.update_task_reward_v2(gathered_task, gathered_reward_per_func, gathered_response_modes)
        except:
            print('failing to update the reward metrics')

        # calculate the grouped completion lengths
        if self.group_num > 1:
            try:
                completion_length_per_sample = completion_mask.sum(1).reshape(-1)
                gathered_length_per_sample = self.accelerator.gather_for_metrics(completion_length_per_sample)
                mode_to_length = defaultdict(list)
                for i, t_m in enumerate(gathered_response_modes):
                    mode_to_length[t_m].append(gathered_length_per_sample[i].item())
                for t_m, v in mode_to_length.items():
                    self._metrics["completion_length/{}".format(t_m)].append(sum(v)/len(v))
            except:
                print("failing to update the completion_length by group")
        
        reward_per_func = gathered_reward_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__

            if reward_func_name == 'myformat_reward':
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item()*10) # since format-reward * 0.1 before.
            else:
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def update_task_reward(self, task, gathered_reward_per_function):
        '''
        task: List of task names, e.g., ['all', 'math', 'reasoning']
        gathered_reward_per_function: tensor of 
        '''
        grouped_gather_metrics = gathered_reward_per_function[:, 0].view(len(task), self.group_num, -1).mean(dim=-1)
        task2metrics = defaultdict(list)
        for i, t in enumerate(task):
            task2metrics[t].append(grouped_gather_metrics[i])
        assert self.reward_funcs[0].__name__ in ['accuracy_reward', "reward_distributer", "general_task_reward"], "The first reward function should be accuracy_reward or reward_distributer (general_task_reward)"
        for t,v in task2metrics.items():
            assert t in self.tasks, f"Task {t} not in the list of provided tasks: {self.tasks}"
            current_task_metrics = torch.stack(v, dim=0).mean(dim=0)  # average the metrics across all processes
            for j, t_m in enumerate(self.think_mode):
                self.task_metrics[t][t_m] = current_task_metrics[j].item()

    def update_task_reward_v2(self, task, gathered_reward_per_function, gathered_response_modes):
        '''
        task: List of task names, e.g., ['all', 'math', 'reasoning']
        gathered_reward_per_function: tensor of 
        '''
        assert len(task) == len(gathered_response_modes)
        grouped_gather_metrics = gathered_reward_per_function[:, 0].view(-1)
        task_mode_metrics = defaultdict(list)
        for i, t in enumerate(task):
            c_mode = gathered_response_modes[i]
            task_mode_metrics[(t, c_mode)].append(grouped_gather_metrics[i].item())
        assert self.reward_funcs[0].__name__ in ['accuracy_reward', "reward_distributer", "general_task_reward"], "The first reward function should be accuracy_reward or reward_distributer (general_task_reward)"
        for k,v in task_mode_metrics.items():
            t, t_m = k
            assert t in self.tasks, f"Task {t} not in the list of provided tasks: {self.tasks}"
            current_task_mode_metrics = sum(v) / len(v)
            self.task_metrics[t][t_m] = current_task_mode_metrics
            
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        metrics_per_task = {f"{task}/{key}": value for task, met in self.task_metrics.items() for key, value in met.items()}
        logs = {**logs, **metrics, **metrics_per_task}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
    
    def adaptive_metrics(self, completions):
        completions = [completion[0]["content"] for completion in completions]
        def type_map(content):
            if content.startswith('<grounding>'):
                return "grounding"
            elif content.startswith('<text>'):
                return "text"
            elif content.startswith('</think>'):
                return "non-thinking"
            else:
                return "others"
        completion_types = [type_map(c) for c in completions]
        types_counter = Counter(completion_types)
        return len([c for c in completion_types if c=="grounding"]) / len(completions), types_counter.most_common(1)[0][1] / len(completions)

    def completion_to_mode(self, completions):
        completions = [completion[0]["content"] for completion in completions]
        modes = []
        for c in completions:
            if self.group_prefix is None:
                c_mode = "mode"
            else:
                for i,prefix in enumerate(self.group_prefix):
                    if c.startswith(prefix):
                        break
                c_mode = self.think_mode[i]
            modes.append(c_mode)
        return modes
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
