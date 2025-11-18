import torch
import os

eval_and_logging_steps = 10
IMG_TOKEN_NUM = 8
ALL_IMG_TOKENS = [f"[IMG{i}]" for i in range(IMG_TOKEN_NUM)]
ALL_IMG_TOKENS_STR = '<ImageHere>' # "<Img><ImageHere></Img>"# "".join(ALL_IMG_TOKENS)

# Location related tokens
LOC_TOKEN_NUM = 256
ALL_LOC_TOKENS = ["[LOC{}]".format(i+1) for i in range(LOC_TOKEN_NUM)]

USE_PREFIX_TUNING = False
USE_LORA = False
USE_CFG = True
IGNORE_TOKEN_ID = -100
IMAGE_TOKEN_INDEX = -200


PRECISION = torch.bfloat16
TRAINABLE_PRECISION = torch.float32

IGNORE_INDEX = -100
DEFAULT_GRD_TOKEN = "<grounding>"
DEFAULT_BOP_TOKEN = ""              # begin of phrase modified from 3/11
DEFAULT_EOP_TOKEN = ""              # end of phrase modified from 3/11
DEFAULT_BOT_TOKEN = "<|start_of_thought|>"  # begin of thought
DEFAULT_EOT_TOKEN = "<|end_of_thought|>"    # end of thought
DEFAULT_BOO_TOKEN = "<obj>"         # begin of object
DEFAULT_EOO_TOKEN = "</obj>"        # end of object
DEFAULT_BOC_TOKEN = "<coor>"        # begin of coordinates
DEFAULT_EOC_TOKEN = "</coor>"       # end of coordinates
DEFAULT_SEP_TOKEN =  " and"         # "<delim>" modified from 3/11
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# begin of image
DEFAULT_BOI_TOKEN = "<Img>"
# end of image
DEFAULT_EOI_TOKEN = '</Img>'
# default image token
DEFAULT_IMG_TOKEN = '<ImageHere>'
COT_ACTIVATION = 'Answer the question and include the reasoning proess. Locate key objects and provide bounding boxes in your thoughts.'
COT_ACTIVATION_TXT = 'Answer the question and include the reasoning proess.'

# placeholder for object features
OBJECT_PLACEHOLDER = "<|object_placeholder|>"

R1_SYSTEM_PROMPT_TXT = "You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

R1_SYSTEM_PROMPT_GRD = "You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning process, please locate key objects and provide their bbox coordinates in the image."

R1_SYSTEM_PROMPT_BASE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>"
)


R1_SYSTEM_PROMPT_RAW = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>"
)

R1_SYSTEM_PROMPT_ADAPT = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have two modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode."
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

# R1_SYSTEM_PROMPT_ADAPT_v3 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. You have two modes of thinking, and you should choose the appropriate one based on the question:"
# "\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, structure your response in the following format: <grounding> <think> reasoning process here </think> <answer> answer here </answer>."
# "\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, structure your response in the following format: <text> <think> reasoning process here </think> <answer> answer here </answer>."
# "\nChoose the mode that best fits the task, and structure your response accordingly."
# )

R1_SYSTEM_PROMPT_ADAPT_v2 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have two modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, begin your response with the tag <text>."
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

R1_SYSTEM_PROMPT_ADAPT_v3 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have three modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, begin your response with the tag <text>."
"\n3. Direct Answer: Use this mode when you believe the question is simple enough to answer directly, skipping the reasoning process. When using this mode, reply in the following format </think> <answer> answer here </answer>"
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

R1_SYSTEM_PROMPT_ADAPT_v2_reverse = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have two modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, begin your response with the tag <text>."
"\n2. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

R1_SYSTEM_PROMPT_GRD_v2 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
"In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object [x1, y1, x2, y2]`. For example, \"<think> The cat [50, 73, 152, 145] is lying on the bed [0, 51, 200, 200]. </think>\""
)

R1_SYSTEM_PROMPT_ADAPT_v4 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have two modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object [x1, y1, x2, y2]`. When using this mode, begin your response with the tag <meta mode=\"grounded reasoning\">."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, begin your response with the tag <meta mode=\"textual reasoning\">."
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

POST_PROMPT_ADAPT_v2 = "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally output the answer according to the question's requirements."

adaptive_mode2prefix = {
    "adapt_v2": {"text": "<text>", "grounding": "<grounding>"},
    "adapt_v4": {"text": '<meta mode="textual reasoning">', "grounding": '<meta mode="grounded reasoning">'}, 
}