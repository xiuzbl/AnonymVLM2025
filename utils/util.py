
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import logging
import textwrap
import importlib

from PIL import Image
from base64 import b64encode, b64decode
from colorama import Fore, Style
from prettytable import PrettyTable
from safetensors import safe_open

import transformers
import torch
import torch.distributed as dist


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_safetensor(path):
    tmp_dict = {}
    with safe_open(path, framework='pt', device=0) as f:
        for k in f.keys():
            tmp_dict[k] = f.get_tensor(k)
    return tmp_dict


def sanitize_filename(filename):
    return re.sub('[^0-9a-zA-Z]+', '_', filename)


def plot_images_and_text(predicted_image1, predicted_image2, groundtruth_image, generated_text, gt_text, save_dir, task_name, input_texts, input_images):
    task_path = os.path.join(save_dir, task_name)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    max_width = 50  # adjust this value based on your needs

    fig, ax = plt.subplots()
    ax.imshow(predicted_image1)
    generated_text = generated_text.replace("###", "").replace("[IMG0]", "")
    wrapped_generated_text = textwrap.fill(generated_text, max_width)
    ax.set_title(wrapped_generated_text, pad=20)
    ax.axis('off')
    plt.savefig(os.path.join(task_path, f"generated.jpg"), bbox_inches='tight')
    plt.close(fig)

    gt_text = gt_text.replace("$", "\$")
    wrapped_gt = textwrap.fill(gt_text, max_width)
    if predicted_image2 is not None:
        fig, ax = plt.subplots()
        ax.imshow(predicted_image2)
        ax.set_title(wrapped_gt, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(task_path, f"sd_baseline.jpg"), bbox_inches='tight')
        plt.close(fig)

    if groundtruth_image is not None:
        fig, ax = plt.subplots()
        groundtruth_image = groundtruth_image.float().cpu().numpy().squeeze()
        groundtruth_image = np.transpose(groundtruth_image, (1, 2, 0))
        groundtruth_image = np.uint8(groundtruth_image*255)
        ax.imshow(groundtruth_image)
        ax.set_title(wrapped_gt, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(task_path, f"gt.jpg"), bbox_inches='tight')
        plt.close(fig)

    if len(input_texts):
        max_width = 30
        length = len(input_texts)
        if length > 1:
            fig, ax = plt.subplots(1, length, figsize=(10*length, 10))
            for i in range(length):
                if i < len(input_images):
                    ax[i].imshow(input_images[i])
                    ax[i].set_title(textwrap.fill(input_texts[i], max_width), fontsize=28)
                    ax[i].axis('off')
                else:
                    ax[i].text(0.5, 0.5, textwrap.fill(input_texts[i], max_width), horizontalalignment='center', verticalalignment='center', fontsize=28)
                    ax[i].axis('off')
        else:
            fig, ax = plt.subplots()
            ax.imshow(input_images[0])
            ax.set_title(textwrap.fill(input_texts[0], max_width), fontsize=28)
            ax.axis('off')
        plt.savefig(os.path.join(task_path, f"input.jpg"), bbox_inches='tight')
        plt.close(fig)

    return None

def instantiate_from_config(config, inference = False, reload=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"], reload=reload)(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def print_trainable_params(model, color="cyan"):
    trainable_params = [k for k, v in model.named_parameters() if v.requires_grad]
    trainable_params_group = {}
    for para in trainable_params:
        layer_num = re.findall(r'layers.(\d+)\.', para)
        block_num = re.findall(r'blocks.(\d+)\.', para)
        if layer_num:
            cur_layer = int(layer_num[0])
            if para.replace('layers.'+layer_num[0],'layers.*') not in trainable_params_group:
                trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]
            elif cur_layer > int(trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')]):
                trainable_params_group[para.replace('layers.'+layer_num[0],'layers.*')] = layer_num[0]      
        elif block_num:
            cur_block = int(block_num[0])
            if para.replace('blocks.'+block_num[0],'blocks.*') not in trainable_params_group:
                trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')] = block_num[0]
            elif cur_block > int(trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')]):
                trainable_params_group[para.replace('blocks.'+block_num[0],'blocks.*')] = block_num[0]
        else:
            trainable_params_group[para] = '0'
    
    table = PrettyTable(['Parameter Name','Max Layer Number'])
    for key in trainable_params_group.keys():
        table.add_row([key, str(int(trainable_params_group[key])+1)])
    
    rank_0_print(table, color=color)
    total_num = sum([v.numel() for k,v in model.named_parameters()])
    trainable_num = sum([v.numel() for k,v in model.named_parameters() if v.requires_grad])
    rank_0_print('Total: {:.2f}M'.format(total_num/1e6), ' Trainable: {:.2f}M'.format(trainable_num/1e6), color=color)


def rank_0_print(*values: object, color="", sep=" ", end="\n", reset=True):
    color_map = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "black": Fore.BLACK,
        "": ""
    }
    msg = color_map[color]
    if dist.is_initialized():
        msg += f"[RANK {dist.get_rank()}]"
        if dist.get_rank() == 0: 
            print(msg, *values, sep=sep, end=end)
    else: 
        msg_str = f"{msg}" + sep.join([str(v) for v in values])
        print(msg_str, sep=sep, end=end)
    
    if color and reset: print(Style.RESET_ALL, end="")
    sys.stdout.flush()


def byte2image(byte_data):
    """
    convert byte to PIL image
    """
    if isinstance(byte_data, str):
        byte_data = b64decode(byte_data)
    image = Image.open(io.BytesIO(byte_data))
    return image


def print_memory_usage(tag, device="cuda:0", padding_length=30):
    """
    打印显存使用情况，带标记信息
    :param tag: 当前代码块的标记信息
    """
    tag = " " + tag + " "*(max(padding_length - 1 - len(tag), 0)) # padding

    if not torch.cuda.is_available():
        print(f"[{Fore.RED}ERROR{Style.RESET_ALL}] CUDA 不可用，无法获取显存信息")
        return

    rank_str = f"[RANK {dist.get_rank()}] " if dist.is_initialized() else ""
    # 获取显存信息
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # 已分配显存 (GB)
    reserved = torch.cuda.memory_reserved(device) / 1024**3    # 预留显存 (GB)

    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # 总显存 (GB)
    free_memory = total_memory - reserved  # 剩余可用显存 (GB)
    usage_percentage = (allocated / total_memory) * 100  # 显存使用率 (%)

    # Use yellow color for tag
    print(f"{rank_str}[{Fore.YELLOW}{tag}{Style.RESET_ALL}] "
          f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Free: {free_memory:.2f} GB | "
          f"Usage: {usage_percentage:.2f}%")
    

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def zero3_parameter_print(m):
    print(m.weight)