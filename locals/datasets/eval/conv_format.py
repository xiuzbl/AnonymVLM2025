from datasets import load_dataset
from torch.utils.data import Dataset
import json
from typing import Any
import os, math
from constants import COT_ACTIVATION, ALL_IMG_TOKENS_STR
from PIL import Image
import pandas as pd
from utils.util import byte2image

def xywh_to_xyxy(bbox):
    new_bbox = [c for c in bbox]
    new_bbox[2] = bbox[0] + bbox[2]
    new_bbox[3] = bbox[1] + bbox[3]
    return new_bbox

class MMStarDataset(Dataset):
    def __init__(
        self,
        path: str,
        append_cot: bool=False,
        system_prompt: str=None
    ):
        self.path = path
        self.meta = load_dataset(path, 'val')
        self.append_cot = append_cot
        self.system_prompt = system_prompt
        self.additional_prompt = None

    def __len__(self) -> int:
        return len(self.meta['val'])
    
    def getlabel(self, i):
        return self.meta['val'][i]['answer']

    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta['val'][i]
        image = item['image']
        if self.append_cot:
            question = "<grounding>\n" + item['question'] + ' ' + COT_ACTIVATION
        else:
            question = item['question']
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "mmstar_{}".format(i)}

   
class VStarDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str,
        append_cot: bool=False,
        system_prompt: str=None,
    ):
        self.path = path
        self.meta = [json.loads(line) for line in open(path)]
        self.append_cot = append_cot
        self.system_prompt = system_prompt
        self.additional_prompt = None
        self.image_dir = image_dir
        self.option_letters = ['A', 'B', 'C', 'D']

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['label']

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        question = item['text']
        image = Image.open(os.path.join(self.image_dir, item['image']))
        question = question.replace("Answer with the option's letter from the given choices directly.", "")
        # the question has already included options
        if self.append_cot:
            question = "<grounding>\n" + question + ' ' + COT_ACTIVATION
        else:
            question = question
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "vstar_{}".format(i)}
    
class SpatialScoreDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        base_dir: str=None,
        append_cot: bool=False,
        system_prompt: str=None,
    ):
        self.meta = json.load(open(meta_path))
        self.base_dir = base_dir
        self.append_cot = append_cot
        self.system_prompt = system_prompt
        self.additional_prompt = None

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']

    def get_image(self, image_link):
        # test_dir = "/data/oss_bucket_0/mllm_dataset/public_datasets/COCO/images/"
        base_img_fn = '/'.join(image_link.split('/')[-2:])
        return Image.open(os.path.join(self.image_dir, base_img_fn)).convert('RGB')

    def prompt_process(self, sample):
        dataset_name = sample.get('source', 'unknown')
        # Define common system prompt for multiple choice questions
        if dataset_name in ["cvbench", "MMIU", "BLINK", "3DSRBench", "MMVP"]:
            assistant_prompt = (
                "**Please select the most appropriate answer from options (A), (B), (C), (D), (E), or (F).**\n"
                "**Respond ONLY with the letter and its parentheses, for example: (A)**\n\nQuestion: "
            )
        elif dataset_name in ["spatialbench", "VSR-ZeroShot", "VSR-Random", "SpatialSense", "VSI-Bench_8", "RealWorldQA"]:
            assistant_prompt = ("**Answer concisely with a single word, number, or option (e.g., yes, no, 5, 2.2, A).**\n\nQuestion: ")
        elif dataset_name in ["QSpatialBench-Plus", "QSpatialBench-ScanNet"]:
            assistant_prompt = (
                "You will be provided with a question and a 2D image. The question involves measuring the precise distance in 3D space through a 2D image. You will answer the question by providing a numeric answer consisting of a scalar and a distance unit in the format of **\scalar{scalar} \distance_unit{distance unit}** at the end of your response.\n"
                "Let's think step by step and start by finding good reference objects or object parts in the image.\n\n"
                "Question: "
            )
        elif dataset_name == "VGBench":
            if sample.get('question_type') == 'open-ended':
                assistant_prompt = (
                "You will be provided with a question and a 2D image. The question involves measuring the precise distance in 3D space through a 2D image. You will answer the question by providing a numeric answer consisting of a scalar and a distance unit in the format of **\scalar{scalar} \distance_unit{distance unit}** at the end of your response.\n"
                "Let's think step by step and start by finding good reference objects or object parts in the image.\n\n"
                "Question: "
            )
            else:
                assistant_prompt = ("**Answer concisely with a single word, number, or option (e.g., yes, no, 5, 2.2, A).**\n\nQuestion: ")
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
        
        # Process images - support multiple images
        images = [Image.open(os.path.join(self.base_dir,img_path)).convert('RGB') for img_path in sample['img_paths']]
        return images, assistant_prompt
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        images, assistant_prompt = self.prompt_process(item)
        
        question = item['question']
        if self.append_cot:
            question = "<grounding>\n" + question + ' ' + COT_ACTIVATION
        else:
            question = question
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        image_content = [{"type": "image", "image": img} for img in images]

        messages = [
            {"role": "user", "content": [{"type": "text", "text": assistant_prompt}] + image_content + [{"type": "text", "text": question}]}
        ]

        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "spatialscore_{}".format(i)}


class POPEHFDataset(Dataset):
    # a dataset class for POPE in lmms-eval hf format
    def __init__(
        self,
        meta_path: str,
        split: str=None,
        append_cot: bool=False,
        system_prompt: str=None,
    ):
        self.meta = load_dataset(meta_path)['test']
        if split is not None:
            self.meta = self.meta.filter(lambda x: x['category']==split, keep_in_memory=True)
            assert len(self.meta) > 0, "no data is in the '{}' split".format(split)
        self.append_cot = append_cot
        self.system_prompt = system_prompt
        self.additional_prompt = None

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        image = item['image']
        
        question = item['question']
        if self.append_cot:
            question = "<grounding>\n" + question + ' ' + COT_ACTIVATION
        else:
            question = question
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "pope_{}".format(i)}

class MathVistaDataset(Dataset):
    # a dataset class for POPE in lmms-eval hf format
    def __init__(
        self,
        meta_path: str,
        split: str=None,
        append_cot: bool=False,
        system_prompt: str=None,
        image_query_sep: str="",
    ):
        self.meta = load_dataset(meta_path)[split]
        self.append_cot = append_cot
        self.system_prompt = system_prompt
        self.additional_prompt = None
        self.image_query_sep = image_query_sep

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        image = item['decoded_image']
        
        question = item['query']
        if self.append_cot:
            question = "<grounding>\n" + question + ' ' + COT_ACTIVATION
        else:
            question = question
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": self.image_query_sep + question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "mathvista_{}".format(i)}


class MathVisionDataset(Dataset):
    # a dataset class for POPE in lmms-eval hf format
    def __init__(
        self,
        meta_path: str,
        split: str=None,
        system_prompt: str=None,
        image_query_sep: str="",
        detailed_prompt: bool=False,
        hint_first: bool=False,
    ):
        self.meta = load_dataset(meta_path)[split]
        self.system_prompt = system_prompt
        self.additional_prompt = None
        self.option_letters = [chr(ord("A")+i) for i in range(26)]
        self.detailed_prompt = detailed_prompt
        self.hint_first = hint_first

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']

    def get_gt_type(self, answer):
        try:
            int(answer)
            return "integer"
        except:
            try:
                float(answer)
                return "float"
            except:
                return "others"

    
    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        image = item['decoded_image']
        
        question = item['question']
        if item['options']:
            options = ['{}. {}'.format(self.option_letters[i], opt) for i, opt in enumerate(item['options'])]
            question = question + '\nOptions: ' + '; '.join(options) + '. '
            post_instruct = "Please provide your final answer using the correct option letter, e.g., A, B, C, D, at the end."
        elif self.detailed_prompt:
            # make hint-prompt based on the ground-truth type
            gt_type = self.get_gt_type(item['answer'])
            if gt_type == "integer":
                post_instruct = "Please provide your final answer using an integer, e.g., 1, 2, 3, at the end."
            elif gt_type == "float":
                post_instruct = "Please provide your final answer using floating-point number, e.g., 1.2, 1.3, 1.4, at the end."
            else:
                post_instruct = 'Please provide your final answer using a number or mathematical expression.'
        else:
            post_instruct = 'Please provide your final answer using a single number.'
        
        # use different formats of answer type instruct
        if self.hint_first:
            # use the Hint: xxx\nQuestion: yyy format similar to MathVista
            question = "Hint: {}\nQuestion: {}".format(post_instruct, question)
        else:
            question = question + post_instruct
        
        if "<image1>" in question and "<image2>" not in question:
            question = question.replace("<image1>", "").strip()
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "mathvision_{}".format(i)}

class MathVerseDataset(Dataset):
    # a dataset class for POPE in lmms-eval hf format
    def __init__(
        self,
        meta_path: str,
        split: str="testmini",
        categories: str="all",
        system_prompt: str=None,
        image_query_sep: str="",
        detailed_prompt: bool=False,
        hint_first: bool=False,
        direct_answer: bool=False,
    ):
        self.meta = load_dataset(meta_path, split)[split]
        # filter out samples that are not desired
        if categories != "all":
            categories = set(categories.split(","))
            self.meta = self.meta.filter(lambda item: item['problem_version'] in categories, keep_in_memory=True)
            assert len(self.meta), "the provided categories: {} are not valid".format(','.join([c for c in categories]))
        self.system_prompt = system_prompt
        self.additional_prompt = None
        self.option_letters = [chr(ord("A")+i) for i in range(26)]
        self.detailed_prompt = detailed_prompt
        self.hint_first = hint_first
        self.direct_answer = direct_answer

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        image = item['image']
        
        if self.direct_answer:
            question = item['query_wo']
        else:
            question = item['query_cot']
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "mathverse_{}".format(i)}


class WeMathDataset(Dataset):
    # a dataset class for POPE in lmms-eval hf format
    def __init__(
        self,
        meta_path: str,
        split: str="testmini",
        categories: str="all",
        system_prompt: str=None,
        image_query_sep: str="",
    ):
        self.meta = load_dataset(meta_path)[split]
        # filter out samples that are not desired
        self.system_prompt = system_prompt
        self.additional_prompt = None
        self.option_letters = [chr(ord("A")+i) for i in range(26)]

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta[i]
        image = item['image_path']
        
        question = "Question: {}\nChoices: {}".format(item['question'], item['option'])
        if self.additional_prompt is not None:
            question = question + ' ' + self.additional_prompt
        messages = [{'role': "user", "content":[
            {'type': 'image', "image": image},
            {'type': "text", "text": question}
        ]},
        ]
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        return {'conv': messages, "item_id": "wemath_{}".format(i)}