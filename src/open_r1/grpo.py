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
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from math_verify import parse, verify
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from trainer.adaptive_grpo_trainer import Qwen2VLAdaptGRPOTrainer
from trainer.adaptive_grpo_trainer_vllm import Qwen2VLAdaptGRPOVLLMTrainer
from util import instantiate_from_config
from processor.qwen25_processor import Qwen2_5_VLProcProcessor
from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor
import transformers
from torch.utils.data import ConcatDataset
from reward_funcs.basic_reward import accuracy_reward, format_reward, count_reward, general_task_reward, grounding_encourage_reward, grounding_encourage_reward_zero
from reward_funcs.group_preference import group_pref_win_probability
from prompt_constant import *
import torch
from pathlib import Path

# os.environ["WANDB_MODE"] = "offline"

import json
import wandb

# wandb.init(project="R1-multimodal", name="Qwen2_5_7B_R1-multimodal")


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: Optional[str] = field(
        default="adaptive,format",
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'count', 'adaptive'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    model_type: Optional[str] = field(
        default="base",
        metadata={"help": "Model type, please use base or vocot"},
    )
    data_config: Optional[str] = field(
        default=None,
        metadata={"help": "optional data config"}
    )
    count_reward: Optional[bool] = field(
        default=False
    )
    vllm: Optional[bool] = field(
        default=False
    )
    group_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Group prefix for the model"},
    )
    use_cache_for_generation: Optional[bool] = field(
        default=False
    )
    rl_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "RL tasks to run, e.g., 'math,qa'"},
    )
    post_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "indicate which post prompt to use (leave it blank to drop post prompt)"}
    )
    freeze_vis_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to freeze the visual encoder"}
    )
    freeze_backbone: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to freeze the backbone"}
    )
    dynamic_mode_strategy: Optional[str] = field(
        default="disable",
        metadata={"help": "the strategy used for dynamically adjust the adaptive mode"}
    )
    generation_temperature: Optional[float] = field(
        default=0.9,
        metadata={"help": "the temperature used during rollout generation, default to 0.9"}
    )
    adaptive_strategy: Optional[str] = field(
        default="in_batch",
        metadata={"help": "the strategy used to apply adaptive training, currently random and in_batch are supported"}
    )
    advantage_strategy: Optional[str] = field(
        default="default",
        metadata={"help": "the strategy used to compute the advantage, valid values: default, v2"}
    )
    average_by_token: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to aggregate the loss by token-wise"}
    )

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "count": count_reward,
    "adaptive": general_task_reward,
    "grd_encourage": grounding_encourage_reward,
    "grd_encourage_zero": grounding_encourage_reward_zero,
    "group_pref_win_probability": group_pref_win_probability,
}


def make_dataset(script_args):
    # add system prompt
    if script_args.model_type == "text":
        sys_prompt = SYSTEM_PROMPT_TXT
    elif script_args.model_type == "base":
        sys_prompt = SYSTEM_PROMPT_BASE
    elif script_args.model_type == "vocot":
        sys_prompt = SYSTEM_PROMPT_VOCOT
    elif script_args.model_type == "adaptive":
        sys_prompt = SYSTEM_PROMPT_ADAPT
    elif script_args.model_type == "adaptive_v2":
        sys_prompt = R1_SYSTEM_PROMPT_ADAPT_v2
    elif script_args.model_type == "adaptive_v3":
        sys_prompt = R1_SYSTEM_PROMPT_ADAPT_v3
    elif script_args.model_type == "grd_v2":
        sys_prompt = R1_SYSTEM_PROMPT_GRD_v2
    else:
        raise NotImplementedError
    
    # add post prompt
    if script_args.post_prompt is not None:
        if script_args.post_prompt == "P4":
            post_prompt = POST_PROMPT_P4
        else:
            raise NotImplementedError
    else:
        post_prompt = None
    if script_args.data_config is None:
        # Load the dataset
        dataset = load_dataset('parquet', data_files=script_args.dataset_name)
        # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": json.dumps([
                    {"role": "system", "content": SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": example["problem"]},
                ])
            }

        def make_conversation_image(example):
            return {
                "prompt": json.dumps([
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "file://" + example['image_path']},
                            {"type": "text", "text": example["problem"]},
                        ],
                    },
                ]),
                "question_format": "math"
            }

        if "image" in dataset[script_args.dataset_train_split].features:
            dataset = dataset.map(make_conversation_image)
        else:
            dataset = dataset.map(make_conversation)
            dataset = dataset.remove_columns("messages")
    else:
        from omegaconf import OmegaConf
        data_cfg = OmegaConf.load(script_args.data_config)
        if 'train' in data_cfg:
            # gather multiple datasets
            datasets = []
            for k,v in data_cfg['train'].items():
                tmp_ds = instantiate_from_config(v)
                assert hasattr(tmp_ds, "system_prompt"), "dataset {} should have system_prompt to support training".format(k)
                tmp_ds.system_prompt = sys_prompt
                tmp_ds.post_prompt = post_prompt
                datasets.append(tmp_ds)
            dataset = {'train': ConcatDataset(datasets)}
        else:
            ds = instantiate_from_config(data_cfg[0])
            ds.system_prompt = sys_prompt
            ds.post_prompt = post_prompt
            dataset = {'train': ds}
    return dataset

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs.split(',')]

    # prepare the dataset
    dataset = make_dataset(script_args)

    import random
    item = dataset['train'][random.randint(0, len(dataset['train']))]
    print('sample case:')
    print(item)

    # check the grouped prefix setup
    additional_kwargs = {"freeze_vis_encoder": script_args.freeze_vis_encoder, "freeze_backbone": script_args.freeze_backbone}
    if script_args.group_prefix is not None:
        group_prefix = script_args.group_prefix.split(',')
        # we need to ensure the **FIRST** prefix is an empty string
        group_prefix = [g for g in group_prefix if g != '']
        # group_prefix = [""] + group_prefix
        additional_kwargs["group_prefix"] = group_prefix
        additional_kwargs['processing_class'] = Qwen2_5_VLProcProcessor
    else:
        additional_kwargs['processing_class'] = Qwen2_5_VLProcessor
    
    trainer_cls = Qwen2VLAdaptGRPOTrainer if not script_args.vllm else Qwen2VLAdaptGRPOVLLMTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        use_cache_generation=script_args.use_cache_for_generation,
        tasks=script_args.rl_tasks.split(',') if script_args.rl_tasks is not None else None,
        dynamic_mode_strategy=script_args.dynamic_mode_strategy,
        generation_temperature=script_args.generation_temperature,
        adaptive_strategy=script_args.adaptive_strategy,
        advantage_strategy=script_args.advantage_strategy,
        average_by_token=script_args.average_by_token,
        **additional_kwargs # for compatibility with non-adaptive GRPO
    )

    # Train and push the model to the Hub
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        # Lora model is not support this resume branch, make sure your lora out_dir is empty.
        print('resume from checkpoint')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # training_args.save_steps = 100  # Save checkpoint every 100 steps
    print('training_args:\n', training_args)
    print('script_args:\n', script_args)
    print('model_args:\n', model_args)
    print('transformers version', transformers.__version__)
    main(script_args, training_args, model_args)
