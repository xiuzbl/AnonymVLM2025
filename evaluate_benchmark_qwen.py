from genericpath import samestat
from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from omegaconf import OmegaConf
from utils.util import instantiate_from_config
from transformers import LlamaTokenizer, AutoTokenizer, Qwen2_5_VLProcessor
import transformers
from peft import PeftConfig, PeftModel
from argparse import ArgumentParser
import os
import torch.distributed as dist
from utils.logger import setup_logger
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig
import json
import tqdm
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

def rank0_print(args, res):
    if args.local_rank==0 or args.local_rank == -1:
        print(res)

def get_output_name(args, mid_output=True):
    if mid_output:
        return os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, args.local_rank))
    else:
        return os.path.join(args.output_dir, 
                            '{}.json'.format(args.dataset_name))

def get_all_output_names(args):
    return [os.path.join(args.output_dir, 
                            '{}_rank{}.json'.format(args.dataset_name, r)) for r in range(args.n_gpus)]

def naive_collate_fn(batches):
    assert len(batches) == 1, "only support batch_size=1!"
    return batches[0]

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/mnt/bn/yangmin-priv/luoruipu/checkpoints/Edit-gpt-4-emu-instruct-test/')
    parser.add_argument('--eval_data', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='plain')
    parser.add_argument('--sub_sample', type=int, default=-1)
    parser.add_argument('--min_pixels', type=int, default=None)
    parser.add_argument('--max_pixels', type=int, default=None)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument("--nums_generation", type=int, default=1)
    # quiet-vocot-related parameters
    parser.add_argument('--n_passes', type=int, default=1)
    parser.add_argument('--n_ahead_talk', type=int, default=1)
    parser.add_argument('--use_original_mode', action='store_true')
    # specific prompt
    parser.add_argument('--additional_prompt', type=str, default=None)
    parser.add_argument('--generation_prefix', type=str, default=None)
    # avoid timeout during barrier
    parser.add_argument('--no_barrier', action='store_true')
    parser.add_argument('--cot_method', type=str, default='prompt')
    parser.add_argument('--base_model', type=str, default='qwen25')
    parser.add_argument('--base_model_dir', type=str, default=None)
    parser.add_argument('--vllm', action='store_true')
    args = parser.parse_args()

    base_config_name = os.path.basename(args.eval_data)
    args.dataset_name = base_config_name[:-5] if base_config_name.endswith('.yaml') else base_config_name
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.n_gpus = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.distributed = False
        args.local_rank = -1
        args.n_gpus = -1
    if not os.path.isdir(args.output_dir) and args.local_rank < 1:
        os.makedirs(args.output_dir)
    rank0_print(args, args)
    global logger
    logger = setup_logger('Evaluation', args.output_dir, args.local_rank)
    logger.info('Evaluating with {} GPUs'.format(args.n_gpus))

    if os.path.exists(get_output_name(args, mid_output=False)):
        print('the results already exist, finished!')
        return
    
    if args.precision == 'bf16':
        args.dtype = torch.bfloat16
    elif args.precision == 'fp16':
        args.dtype = torch.float16

    model_path = args.model_path
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs['min_pixels'] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs['max_pixels'] = args.max_pixels
    if args.base_model_dir is not None and args.base_model_dir != model_path:
        processor = AutoProcessor.from_pretrained(args.base_model_dir, **processor_kwargs)
                                                # min_pixels=args.min_pixels,
                                                # max_pixels=args.max_pixels)
        if args.vllm and args.local_rank in [0, -1]:
            config = AutoConfig.from_pretrained(args.base_model_dir)
            config.save_pretrained(model_path)
            processor.save_pretrained(model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
                                                # min_pixels=args.min_pixels,
                                                # max_pixels=args.max_pixels)
    logger.info('loading models from {}, with the {} mode'.format(model_path, args.model_type))
    if args.vllm:
        model = LLM(model=args.model_path, device=device, distributed_executor_backend="external_launcher", seed=42)
        vllm_sampling_params = SamplingParams(n=args.nums_generation, temperature=args.temperature, max_tokens=args.max_new_tokens)
    elif args.model_type=='plain':
        if args.base_model == "qwen25":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=args.dtype)
        elif args.base_model == "qwen2":
            model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=args.dtype)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if not args.vllm:
        model.eval()
        model.to(device)

    # make the dataloader here!
    config = OmegaConf.load(args.eval_data)
    logger.info('loading dataset as {}'.format(json.dumps(OmegaConf.to_object(config))))
    dataset = instantiate_from_config(config[0])
    if args.sub_sample > 0:
        dataset.meta = dataset.meta[:args.sub_sample]
    # make sure use cot
    dataset.append_cot = args.model_type in ['explicit_cot', 'quiet_cot', 'vocot'] and args.cot_method == "prompt"
    if args.cot_method == "system_grd":
        dataset.system_prompt = R1_SYSTEM_PROMPT_GRD
    elif args.cot_method == "system_grd_v2":
        dataset.system_prompt = R1_SYSTEM_PROMPT_GRD_v2
    elif args.cot_method == "system_txt":
        dataset.system_prompt = R1_SYSTEM_PROMPT_TXT
    elif args.cot_method == "system_base":
        dataset.system_prompt = R1_SYSTEM_PROMPT_BASE
    elif args.cot_method == "system_adapt":
        dataset.system_prompt = R1_SYSTEM_PROMPT_ADAPT
    elif args.cot_method == "system_adapt_v2":
        dataset.system_prompt = R1_SYSTEM_PROMPT_ADAPT_v2
    elif args.cot_method == "system_adapt_v2_reverse":
        dataset.system_prompt = R1_SYSTEM_PROMPT_ADAPT_v2_reverse
    elif args.cot_method == "system_adapt_v3":
        dataset.system_prompt = R1_SYSTEM_PROMPT_ADAPT_v3
    elif args.cot_method == "prompt":
        dataset.direct_answer = True # for special benchmarks that distinguishes prompt
        pass
    else:
        raise NotImplementedError(f"Unknown cot method {args.cot_method}")

    # add special prompt
    if args.additional_prompt is not None:
        assert hasattr(dataset, 'additional_prompt'), "current dataset does not support additional prompt!"
        dataset.additional_prompt = args.additional_prompt

    if args.local_rank <= 0:
        import random
        print('example evaluation data')
        print(dataset[random.randint(0, len(dataset)-1)])

    sampler = SequentialSampler(dataset) if not args.distributed else DistributedSampler(dataset, shuffle=False)
    dl = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=naive_collate_fn)
    logger.info("***** Runing Evaluation *****")
    logger.info("  Num examples = %d", len(dataset))

    current_res = []
    first_show = True
    with torch.inference_mode(), torch.no_grad():
        for batch in tqdm.tqdm(dl, desc='evaluating'):
            try:
                conv = batch['conv']
                image_inputs, __ = process_vision_info(conv)
                query = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                test_image = image_inputs[0]
                image_grid_thw = processor.image_processor(test_image)['image_grid_thw']
                if args.generation_prefix is not None:
                    # add special prefix if required
                    query =  query + args.generation_prefix
                if args.vllm:
                    vllm_inputs = {
                        "prompt": query,
                        "multi_modal_data": {"image": image_inputs}
                    }
                    output = model.generate([vllm_inputs], vllm_sampling_params, use_tqdm=False)
                    if args.nums_generation == 1:
                        response = output[0].outputs[0].text
                    else:
                        response = [o.text for o in output[0].outputs]
                    # response = output[0].outputs[0].text
                else:
                    input_dict = {k:v.to(device) for k,v in processor(text=[query], images=image_inputs, padding=True, return_tensors="pt").items()}
                    input_dict['use_cache'] = args.use_cache
                    if args.temperature == 0:
                        input_dict['do_sample'] = False
                    else:
                        input_dict['temperature'] = args.temperature
                    output = model.generate(**input_dict, max_new_tokens=args.max_new_tokens)
                    response = processor.tokenizer.batch_decode(output)[0]
                if first_show:
                    print("Example query:", query + "\n")
                    print("Example response:", response)
                    first_show = False
            except:
                print('fail to predict {}'.format(batch['item_id']))
                response = 'Error!'
            tmp_dict = {'item_id': batch['item_id'], "question": query, "response": response, "image_size": (int(14*image_grid_thw[0][2]), int(14*image_grid_thw[0][1]))}
            if args.model_type=='quiet_cot':
                if model.cache_hidden_thoughts is not None:
                    tmp_dict['hidden_thoughts'] = model.cache_hidden_thoughts
                    model.cache_hidden_thoughts = []
            if hasattr(dataset, 'getlabel'):
                try:
                    item_id = int(tmp_dict['item_id'].split('_')[-1])
                    tmp_dict['label'] = dataset.getlabel(item_id)
                except:
                    pass
            current_res.append(tmp_dict)
    # remove duplication if necessary in Distributed version
    if args.distributed and len(dataset) % args.n_gpus != 0:
        residual_samples = len(dataset) % args.n_gpus
        if not args.local_rank < residual_samples:
            current_res = current_res[:-1]
    
    with open(get_output_name(args, mid_output=True), 'w') as wf:
        json.dump(current_res, wf)

    print('====Finished From Rank {}====='.format(args.local_rank))

    if args.no_barrier:
        if args.local_rank == 0:
            torch.save(args, get_output_name(args, mid_output=False)[:-4]+'args.bin')
        return
    
    torch.distributed.barrier()
    if args.local_rank == 0 or args.local_rank == -1:
        full_res = []
        for fn in get_all_output_names(args):
            full_res.extend(json.load(open(fn, 'r')))
            os.remove(fn)
        with open(get_output_name(args, mid_output=False), 'w') as wf:
            json.dump(full_res, wf)
        # saving the arguments
        torch.save(args, get_output_name(args, mid_output=False)[:-4]+'args.bin')
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
            

if __name__=='__main__':
    main()