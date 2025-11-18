from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
import os, json
from PIL import Image
import math, random
from copy import deepcopy

class JsonBaseDataset(Dataset):
    def __init__(
        self,
        path,
        system_prompt: str=None,
    ):
        self.path = path
        self.base_dir = os.path.dirname(path)
        self.meta = json.load(open(path))

        self.system_prompt = system_prompt
        self.post_prompt = None
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, i):
        item = deepcopy(self.meta[i])
        image_path = os.path.join(self.base_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        messages = json.loads(item['prompt'])

        # remove the unsed <image token>
        messages[-1]['content'][-1]['text'] = messages[-1]['content'][-1]['text'].replace("<image>", "")
        if self.post_prompt is not None:
            messages[-1]['content'][-1]['text'] = messages[-1]['content'][-1]['text'] + " " + self.post_prompt
        if self.system_prompt is not None:
            messages = [{'role': 'system', "content": self.system_prompt}] + messages
        
        item['prompt'] = json.dumps(messages)
        item['image'] = image
        for unsed_key in ["problem", "image_path"]:
            if unsed_key in item:
                del(item[unsed_key])
        return item
