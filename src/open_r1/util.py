import importlib
import torch
import torch.distributed as dist
from colorama import Fore, Style
import sys
from prettytable import PrettyTable
import re
import base64
from io import BytesIO
from PIL import Image

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


def base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    image_buffer = BytesIO(image_data)
    image = Image.open(image_buffer)
    
    return image
