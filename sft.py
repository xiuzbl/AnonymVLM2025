from typing import Any, Optional, Dict, List, Sequence
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import warnings
import numpy as np

from dataclasses import dataclass, field
from omegaconf import OmegaConf
from functools import partial
from pathlib import Path
from PIL import Image
from collections import defaultdict
import transformers

import torch
torch.backends.cuda.matmul.allow_tf32 = True
from torch.utils.data import ConcatDataset
from lightning.pytorch import seed_everything

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoProcessor,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info, smart_resize
from peft import LoraConfig, TaskType, get_peft_model

from locals.datasets import WrappedDataset
from utils.util import (
    instantiate_from_config, 
    rank_0_print,
    print_trainable_params, 
    get_peft_state_non_lora_maybe_zero_3,
    get_peft_state_maybe_zero_3,
    safe_save_model_for_hf_trainer
)
from constants import OBJECT_PLACEHOLDER
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/mnt/public/Fudan/share/models/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use dynamic cache for the model."},
    )
    # LoRA
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: Optional[str] = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    box_first: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include box before the objects"}
    )
    no_object_tag: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to exclude the special tokens for objects"}
    )
    no_object_feat: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include visual features in the reasoning process"}
    )
    space_between_box: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use space to seperate text and bounding boxes"}
    )
    processor_misc_path: Optional[str] = field(
        default=None,
        metadata={"help": "if provided, use this to load the preprocessor"}
    )
    def __post_init__(self):
        if hasattr(self.lora_target_modules, "__len__") and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]


@dataclass
class DataArguments:
    data_config_path: Optional[str] = field(
        default="config/datasets/quiet_stage0_qwen2vl.yaml", 
        metadata={"help": "Path to the data config file."}
    )
    max_wh_limit: Optional[int] = field(
        default=980,
        metadata={"help": "Maximum width or height of the image"},
    )
    max_pixels: Optional[int] = field(
        default=1280*28*28,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=256*28*28,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    max_seq_length: Optional[int] = field(
        default=-1,
        metadata={"help": "Maximal length for input sequences"}
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
    project_name (`str`, *optional*, defaults to `Qwen2_5_VL_Quiet`):
            The name of the project for wandb logging.
    run_name (`str`, *optional*, defaults to `output_dir`):
            A descriptor for the run. Typically used for [wandb](https://www.wandb.com/). 
            If not specified, will be the same as `output_dir`.
    """
    project_name: Optional[str] = field(
        default='Qwen2_5_VL_Quiet',
        metadata={"help": "The name of the project for wandb logging."}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging."},
    )
    freeze_visual_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the visual encoder."},
    )
    freeze_llm_backbone: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the language backbone."},
    )


def get_normal_model(model_args, training_args):
    model_name_or_path = model_args.model_name_or_path

    if "Qwen2.5-VL" in model_name_or_path:
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "Qwen2-VL" in model_name_or_path:
        raise NotImplementedError 
        model_class = Qwen2VLForConditionalGeneration
    else: raise NotImplementedError(f"Huggingface Model {model_name_or_path} not implemented.")

    if model_args.use_cache and training_args.gradient_checkpointing:
        warnings.warn("Gradient checkpointing is not supported with cache. Disabling gradient_checkpointing.")
        training_args.gradient_checkpointing = False

    model_init_kwargs = {}
    # update config with model_args
    if model_args.attn_implementation: model_init_kwargs["attn_implementation"] = model_args.attn_implementation
    model_init_kwargs["use_cache"] = model_args.use_cache
    torch_dtype = model_args.torch_dtype
    if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None: pass
    elif isinstance(torch_dtype, str):
        model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    else: raise ValueError("Invalid `torch_dtype`. Expected either 'auto' "
        f"or a string representing a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}.")
    
    model = model_class.from_pretrained(model_name_or_path, **model_init_kwargs)

    # Frozen some parameters
    if training_args.freeze_visual_encoder:
        model.visual.requires_grad_(False)
    if training_args.freeze_llm_backbone:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
    
    if model_args.use_peft: # Load PEFT model
        if model_args.lora_target_modules:
            lora_target_modules = model_args.lora_target_modules
        else:
            avoid_keys = ['embed_tokens', 'lm_head', 'visual']
            lora_target_modules = []
            for k,v in model.named_modules():
                if any(mm_keyword in k for mm_keyword in avoid_keys): continue
                elif isinstance(v, torch.nn.Linear): lora_target_modules.append(k)
        lora_config = LoraConfig(
            r=model_args.lora_r,
            alpha=model_args.lora_alpha,
            dropout=model_args.lora_dropout,
            target_modules=lora_target_modules,
            task_type=TaskType[model_args.lora_task_type],
            modules_to_save=model_args.lora_modules_to_save
        )
        model = get_peft_model(model, lora_config)
    else: # Make sure to activate the input embeddings and lm head
        model.get_input_embeddings().weight.requires_grad = True

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    return model


def create_sft_labels(input_ids, processor):
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>") # 151644
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")     # 151645
    assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")   # 77091
    vis_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vis_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    vis_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    labels = torch.ones_like(input_ids) * IGNORE_INDEX
    st_indices, end_indices, is_assistant = [], [], False
    for i in range(input_ids.shape[-1]):
        cur_id = input_ids[0, i]
        next_id = input_ids[0, i+1] if i+1 < input_ids.shape[-1] else None
        if cur_id == im_start_id and next_id and next_id == assistant_id: # 检查是否是 assistant 开头
            st_indices.append(i+3)  # +3 because the seq is "<|im_start|>assistant\n"
            is_assistant = True
        elif cur_id == im_end_id and is_assistant: # 检查是否是 assistant 结尾
            end_indices.append(i+2) # +2 because the seq is "<|im_end|>\n"
            is_assistant = False
    
    for st, end in zip(st_indices, end_indices):
        labels[0, st:end] = input_ids[0, st:end]

    # mask out all vision-related tokens
    for neglect_id in [vis_start_id, vis_end_id, vis_pad_id]:
        labels[labels==neglect_id] = IGNORE_INDEX
    
    return labels

def custom_qwen2_5_preprocess(input_dict:Dict, processor:Qwen2_5_VLProcessor, use_bbox=True, **kwargs):
    messages = input_dict['conversation']

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, __ = process_vision_info(messages)
    output_dict = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    output_dict["labels"] = create_sft_labels(output_dict["input_ids"], processor) # Prepare labels for SFT 

    if 'max_seq_length' in kwargs and kwargs['max_seq_length'] > 0:
        # perform truncation
        output_dict['labels'] = output_dict['labels'][:, :kwargs['max_seq_length']]
        output_dict['input_ids'] = output_dict['input_ids'][:, :kwargs['max_seq_length']]
        output_dict['attention_mask'] = output_dict['attention_mask'][:, :kwargs['max_seq_length']]
    
    return output_dict


def custom_qwen2_preprocess(input_dict:Dict, processor:Qwen2VLProcessor, **kwargs):
    raise NotImplementedError("Preprocess function of Huggingface Model Qwen2-VL not implemented.")


def prepare_dataset(data_config, processor, model_name_or_path, **kwargs):
    if "Qwen2.5-VL" in model_name_or_path:
        preprocessor = partial(custom_qwen2_5_preprocess,
                               processor=processor, **kwargs)
    elif "Qwen2-VL" in model_name_or_path:
        preprocessor = partial(custom_qwen2_preprocess,
                               processor=processor, **kwargs)
    else: raise NotImplementedError(f"Preprocess function of Huggingface Model {model_name_or_path} not implemented.")
    
    train_data = []
    for k, v in data_config["train"].items():
        data = instantiate_from_config(v)
        data = WrappedDataset(k, data, processor=preprocessor)
        train_data.append(data)
    train_dataset = ConcatDataset(train_data)

    if "val" in data_config:
        val_data = []
        for k, v in data_config["val"].items():
            data = instantiate_from_config(v)
            data = WrappedDataset(k, data, processor=preprocessor)
            val_data.append(data)
        val_dataset = ConcatDataset(val_data)
    else: val_dataset = None

    return train_dataset, val_dataset


def format_qwen2vl_example(example, processor):
    input_texts = processor.decode(example["input_ids"][0], skip_special_tokens=False)
    num_image_tokens = input_texts.count("<|image_pad|>")
    input_texts = input_texts.replace("<|image_pad|>", "")
    input_texts = input_texts.replace("<|vision_start|><|vision_end|>", 
                                      f"<|vision_start|>{num_image_tokens}<|vision_end|>")
    return input_texts


class Qwen2VL_SFT_DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        feature_keys = features[0].keys()

        batch = dict()
        for key in feature_keys:
            if key == "input_ids":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key][0] for f in features], 
                    padding_side=self.padding_side, 
                    batch_first=True, padding_value=self.pad_token_id
                ).to(torch.long)
            if key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key][0] for f in features], 
                    padding_side=self.padding_side, 
                    batch_first=True, padding_value=IGNORE_INDEX
                ).to(torch.long)
            if key == "attention_mask":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key][0] for f in features], 
                    padding_side=self.padding_side, 
                    batch_first=True, padding_value=0
                ).to(torch.long)
            if key == "bbox_indices":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features], 
                    padding_side="right", 
                    batch_first=True, padding_value=-1
                ).to(torch.long)
            # Image and Video
            if key in ["pixel_values", "pixel_values_videos"]:
                batch[key] = torch.cat([f[key] for f in features], dim=0)
            if key in ["image_grid_thw", "video_grid_thw"]:
                batch[key] = torch.cat([f[key] for f in features], dim=0).to(torch.long)
            if key == "second_per_grid_ts":
                batch[key] = [f[key] for f in features]
            if key == "item_id":
                batch[key] = [f[key] for f in features]

        return batch


def main():
    seed_everything(42)

    # Parse Arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args_str = " ".join(f"{training_args}".split('\n'))
    rank_0_print(model_args, data_args, training_args_str, color="green", sep="\n\n")

    model = get_normal_model(model_args, training_args)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_args.model_name_or_path if model_args.processor_misc_path is None else model_args.processor_misc_path, 
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels
    )
    print('transformers version', transformers.__version__, "(please ensure using 4.51.3)")
    # Load Data
    data_config = OmegaConf.load(data_args.data_config_path)
    train_data, val_data = prepare_dataset(data_config, processor, model_args.model_name_or_path,
                                           max_wh_limit=data_args.max_wh_limit, use_bbox=True,
                                           box_first=model_args.box_first, no_object_tag=model_args.no_object_tag, space_between_box=model_args.space_between_box,
                                           no_object_feat=model_args.no_object_feat, max_seq_length=data_args.max_seq_length)
    rank_0_print(f"Train Dataset size: {len(train_data)}", color="yellow")
    if val_data: rank_0_print(f"Validation Dataset size: {len(val_data)}", color="yellow")

    rank_0_print("="*100)
    example = train_data[random.randint(0, len(train_data))]
    input_texts = format_qwen2vl_example(example, processor)
    example_infos = ["INPUT_TEXTS: \n{}".format(input_texts)]
    for k, v in example.items():
        if v is None:
            example_infos.append(f"{k.upper()} -- None")
        elif k == "image_grid_thw":
            example_infos.append(f"{k.upper()} -- {v} with shape {v.shape}")
        elif isinstance(v, torch.Tensor):
            example_infos.append(f"{k.upper()} -- {v.shape} >> {v.dtype}")
        elif isinstance(v, str):
            example_infos.append(f"{k.upper()} -- {v}")
        else: raise NotImplementedError(f"Unknown type {type(v)}")
    rank_0_print(*example_infos, sep="\n", color="magenta")
    rank_0_print("="*100)

    custom_data_collator = Qwen2VL_SFT_DataCollator(
        pad_token_id=processor.tokenizer.pad_token_id, padding_side="right")
    
    # 创建Trainer
    print_trainable_params(model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=processor,
        data_collator=custom_data_collator,
    )

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        # Lora model is not support this resume branch, make sure your lora out_dir is empty.
        rank_0_print('resume')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    if model_args.use_peft:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), 'none'
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()

