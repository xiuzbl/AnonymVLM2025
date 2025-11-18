SYSTEM_PROMPT_BASE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>"
)


SYSTEM_PROMPT_VOCOT = "You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. In your reasoning process, please locate key objects and provide their bbox coordinates in the image."

SYSTEM_PROMPT_TXT = "You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."

SYSTEM_PROMPT_ADAPT = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have two modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode."
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

R1_SYSTEM_PROMPT_ADAPT_v2 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have two modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, begin your response with the tag <text>."
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

R1_SYSTEM_PROMPT_GRD_v2 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
"In your reasoning path, identify and locate key objects with their corresponding bbox coordinates in the format `object [x1, y1, x2, y2]`."
)

R1_SYSTEM_PROMPT_ADAPT_v3 = ("You are a helpful assistant. The user asks a question related to an image, you need to solve it. Please first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You have three modes of thinking, and you should choose the appropriate one based on the question:"
"\n1. Grounded Thinking: Use this mode when you need to locate specific objects in the visual input. In your reasoning path, identify key objects and provide their corresponding bounding box coordinates in the format `object[x1, y1, x2, y2]`. When using this mode, begin your response with the tag <grounding>."
"\n2. Normal Thinking: Use this mode for general reasoning based solely on textual thoughts. No object localization or coordinate output is required in this mode. When using this mode, begin your response with the tag <text>."
"\n3. Direct Answer: Use this mode when you believe the question is simple enough to answer directly, skipping the reasoning process. When using this mode, reply in the following format </think> <answer> answer here </answer>"
"\nChoose the mode that best fits the task, and structure your response accordingly."
)

POST_PROMPT_P4 = "Please first select the appropriate reasoning mode based on the question, using <grounding> or <text> to indicate the type, then follow the corresponding format to output the reasoning process, and finally output your answer according to the question's requirements."