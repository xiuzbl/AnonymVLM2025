import json
from torch.utils.data import Dataset
import os

class JsonDataset(Dataset):
    def __init__(
            self,
            path:str,
    ):
        self.path = path
        self.base_dir = os.path.dirname(self.path)
        self.meta = json.load(open(self.path))
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        conv = self.meta[index]
        processed_conv = []
        for msg in conv:
            if msg['role'] in ['system', 'assistant']:
                processed_conv.append(msg)
            else:
                assert msg['role'] == "user"
                processed_msg = {"role": "user", "content": []}
                for c in msg['content']:
                    if c["type"] == "image":
                        c["image"] = os.path.join(self.base_dir, c["image"])
                    processed_msg['content'].append(c)
                processed_conv.append(processed_msg)
        return {"conversation": processed_conv}