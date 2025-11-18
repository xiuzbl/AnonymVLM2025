from math_verify import parse, verify
from datetime import datetime
import re
from .grounding_reward import DetectionVerifier, extract_answer_content
from .grounding_helper import extract_bbox, normalize_bbox_by_real_size

def general_task_reward(completions, solution, **kwargs):
    # distribute the result to different reward based on the question type
    # only support batch_size == 1 now
    question_format = kwargs['question_format'] if 'question_format' in kwargs else "math"
    assert len(set(question_format)) == 1
    if isinstance(question_format, list): question_format = question_format[0]
    if "reward_function" in kwargs:
        # distribute the reward functions based on reward_function
        reward_function = kwargs['reward_function']
        assert len(set(reward_function)) == 1
        if isinstance(reward_function, list): reward_function = reward_function[0]
        if reward_function in ['acc', 'mathverify']:
            return accuracy_reward(completions, solution, **kwargs)
        elif reward_function == "iou":
            return iou_reward(completions, solution, **kwargs)
        else:
            raise NotImplementedError
    if question_format == "count" or question_format == "counting":
        return count_reward(completions, solution, **kwargs)
    elif question_format in ['cv_detection', 'cv_grounding']:
        return detect_reward_minimax(completions, solution, **kwargs)
    else:
        assert question_format in ['puzzle','chart','counting','ocr','math','stem']
        return accuracy_reward(completions, solution, **kwargs)

def extract_answer_from_boxed(content):
    # a function designed for answers that are potentially enclosed by \\boxed 
    if '\\boxed' in content or '\\text' in content:
    # 查找所有被\boxed{}或\text{}包裹的数值
        matches = re.findall(r'\\(?:boxed|text)\{([^}]*)\}', content)
    else:
        matches = None
    if matches:
        # 返回第一个匹配结果，如果没有则返回None
        return matches[0]
    else:
        return content

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    question_format = kwargs['question_format'] if 'question_format' in kwargs else "default"
    if isinstance(question_format, list): question_format = question_format[0]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        if question_format == "ocr":
            content, sol = content.lower(), sol.lower()
        reward = 0.0
        # Try symbolic verification first
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            teacher_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            teacher_answer = teacher_match.group(1).strip() if teacher_match else sol.strip()
            answer = parse(student_answer)
            if len(answer) == 0:
                # unable to parse answer
                answer = parse("\\boxed{"+student_answer+"}")
            parsed_teacher_answer = parse(teacher_answer)
            if len(parsed_teacher_answer) == 0:
                # unable to parse text
                parsed_teacher_answer = parse("\\boxed{"+teacher_answer+"}")
            if float(verify(answer, parsed_teacher_answer)) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                ground_truth = extract_answer_from_boxed(ground_truth)

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = extract_answer_from_boxed(student_answer)

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        # if reward > 0:
        #     think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        #     if think_match:
        #         pass
        #     else:
        #         reward = 0

        rewards.append(reward)
        print(f"------------- {current_time} Accuracy reward for task {question_format}: {reward} -------------\n")
        print(f"Content: {content}\n")
        print(f"Solution: {sol}\n")
    return rewards

def count_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            answer = parse(student_answer)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                ground_truth = extract_answer_from_boxed(ground_truth)

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = extract_answer_from_boxed(student_answer)

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
                else:
                    gt_number = int(re.findall('(\d+)', ground_truth)[0])
                    student_number = int(re.findall('(\d+)', student_answer)[0])
                    diff = abs(gt_number - student_number)
                    reward = 1 / (1+diff)
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        # if reward > 0:
        #     think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        #     if think_match:
        #         pass
        #     else:
        #         reward = 0
        rewards.append(reward)

        print(f"------------- {current_time} Count reward: {reward} -------------\n")
        print(f"Content: {content}\n")
        print(f"Solution: {sol}\n")
    return rewards

def detect_reward_minimax(completions, solution, iou_completeness_weight=0.3, iou_threshold=0.5, **kwargs):
    # prepare the inputs
    assert 'image_grid_thw' in kwargs, "please provide image_grid_thw for bbox rescale!"
    image_grid_thw = kwargs['image_grid_thw']
    contents = [completion[0]["content"] for completion in completions]
    detect_rewards = []
    detect_verifier = DetectionVerifier(is_training=True, step=0, total_steps=100)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # compute the grounding metrics
    for content, sol in zip(contents, solution):
        predict_extract = extract_answer_content(content.strip()).lower()
        answer = extract_answer_content(sol.strip()).lower()

        # for both predict and answer, ignore confidence
        predict_bbox = extract_bbox(predict_extract, ignore_confidence=True)
        answer_bbox = extract_bbox(answer, ignore_confidence=True)

        if answer_bbox is None:
            print(f"Check GT! No bbox found in ground truth: {sol}")
            detect_rewards.append(0.0)
            continue

        if predict_bbox is None:
            format_score = detect_verifier.verify_format(content)
            if format_score == 0.0:
                detect_rewards.append(0.0)
                print(
                    f"Current predict does not include bbox in predict_str: {content}"
                )
                continue
            else:
                print(
                    f"Potential format error! Format score: {format_score}, but no bbox found in predict_str: {content}"
                )
                detect_rewards.append(0.0)
                continue

        predict_bbox = normalize_bbox_by_real_size(
            pred_bboxes=predict_bbox,
            input_width=image_grid_thw[0][2].item()*14,
            input_height=image_grid_thw[0][1].item()*14,
            normalize_size=1000.0,
        )

        # Handle empty predict_bbox after normalization
        if len(predict_bbox) == 0:
            detect_rewards.append(0.0)

        # size_penalty ranges from (0, 1.0], 1.0 when lengths match, approaches 0 as difference increases
        size_penalty_ratio = 0.6
        size_penalty = size_penalty_ratio**(abs(len(predict_bbox) - len(answer_bbox)))

        # ==================iou score=================
        # {'mean_iou_score': mean_iou_score, 'completeness_score': completeness_score, 'precision': 1- false_alarm_rate, 'recall': 1- miss_rate, 'weighted_iou_score': weighted_iou_score}}
        # adjust the iou_threshold
        step_ratio = kwargs['global_step'] / kwargs['max_steps']
        if step_ratio <= 0.1:
            select_iou = 0.5
        elif step_ratio <= 0.25:
            select_iou = 0.5
        else:
            select_iou = 0.5
        reward_res = detect_verifier.calculate_iou_score(predict_bbox, answer_bbox, "greedy_match_by_iou_max_label_first", select_iou, iou_completeness_weight, 1-iou_completeness_weight)
        reward = reward_res['weighted_iou_score']*size_penalty
        detect_rewards.append(reward)
        try:
            print(f"------------- {current_time} grounding reward: {reward} -------------\n")
            print(f"Content: {content}\n")
            print(f"Solution: {sol}\n")
        except:
            print("log writing error")
    return detect_rewards

def strict_non_think_format_reward(content):
    assert content.startswith('</think>') # rollout of thinking modes should be be passed into this function
    direct_answer_pattern_strict = r"^</think>\s*<answer>.*?</answer>$"
    count_1_tags = ['</think>', '<answer>', '</answer>']
    count_0_tags = ['<think>', '<text>', '<grounding>']
    if re.match(direct_answer_pattern_strict, content, re.DOTALL):
        for c1t in count_1_tags:
            if content.count(c1t) != 1:
                return 0.0
        for c0t in count_0_tags:
            if content.count(c0t) != 0:
                return 0.0
        return 1.0
    else:
        return 0.0

def bbox_format_verify(content):
    predict_answer = extract_answer_content(content).strip()
    bbox_pattern = r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]"
    return 1.0 if re.match(bbox_pattern, predict_answer, re.DOTALL) else 0.0


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think> <answer>.*?</answer>$"
    c_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    flexible_pattern = r".*<think>.*?</think>\s*<answer>.*?</answer>"
    # direct answer prompt
    direct_answer_pattern = r"^</think>\s*<answer>.*?</answer>"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(flexible_pattern, content, re.DOTALL) for content in completion_contents]

    # further consider the bounding box format
    question_format = kwargs['question_format'] if 'question_format' in kwargs else "math"
    if isinstance(question_format, list): question_format = question_format[0]
    if question_format == "TT_RefExp":
        detect_matches = []
        for i,match in enumerate(matches):
            content = completion_contents[i]
            if content.startswith('</think>'):
                # non-thinking mode
                non_think_match = strict_non_think_format_reward(content)
                if non_think_match == 0.0:
                    detect_matches.append(0.0)
                else:
                    detect_matches.append(bbox_format_verify(content))
            else:
                # thinking mode
                if not match:
                    detect_matches.append(0.0)
                else:
                    detect_matches.append(bbox_format_verify(content))
        return detect_matches
    elif question_format in ['cv_detection', 'cv_grounding']:
        detect_verifier = DetectionVerifier(is_training=True, step=0, total_steps=100)
        detect_matches = []
        for i,match in enumerate(matches):
            content = completion_contents[i]
            if content.startswith('</think>'):
                # non-thinking mode
                non_think_match = strict_non_think_format_reward(content)
                if non_think_match == 0.0:
                    detect_matches.append(0.0)
                else:
                    detect_matches.append(detect_verifier.verify_format(content))
            else:
                # thinking mode
                if not match:
                    detect_matches.append(0.0)
                else:
                    detect_matches.append(detect_verifier.verify_format(content))
        return detect_matches
    format_rewards = []
    for i, c in enumerate(completion_contents):
        if c.startswith('</think>'):
            format_rewards.append(1.0 if strict_non_think_format_reward(c) else 0.0)
        else:
            format_rewards.append(1.0 if matches[i] else 0.0)
    return format_rewards
    # return [1.0 if match else 0.0 for match in matches]


def grounding_encourage_reward(completions, solution, **kwargs):
    """encourage the grounding ."""
    assert 'accuracy_reward' in kwargs, "the encouragement reward should be based on accuracy"
    accuracy_reward = kwargs['accuracy_reward']
    contents = [completion[0]["content"] for completion in completions]
    bbox_pattern_adaptive = '\[[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+\]'
    rewards = []
    for content, sol, acc in zip(contents, solution, accuracy_reward):
        r = 0
        if content.startswith('<grounding>') and acc==1:
            if re.search(bbox_pattern_adaptive, content):
                r = 0.5
        rewards.append(r)
    return rewards

def grounding_encourage_reward_zero(completions, solution, **kwargs):
    """encourage the grounding in a zero version."""
    def extract_think_content(text):
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else text
    assert 'accuracy_reward' in kwargs, "the encouragement reward should be based on accuracy"
    accuracy_reward = kwargs['accuracy_reward']
    contents = [completion[0]["content"] for completion in completions]
    bbox_pattern_adaptive = '\[[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+\]'
    rewards = []
    for content, sol, acc in zip(contents, solution, accuracy_reward):
        r = 0
        if acc==1:
            think_process = extract_think_content(content)
            if re.search(bbox_pattern_adaptive, think_process):
                r = 1
        rewards.append(r)
    return rewards



def single_iou_reward(content, sol, progress, image_grid_thw):
    input_width=image_grid_thw[0][2].item()*14
    input_height=image_grid_thw[0][1].item()*14
    if progress < 0.1:
        iou_threshold = 0.5
    elif progress < 0.3 and progress >= 0.1:
        iou_threshold = 0.8
    else:
        iou_threshold = 0.95
    
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    
    reward = 0.0
    # Try symbolic verification first
    try:
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        ground_truth = ground_truth.replace("$","")
        ground_truth = eval(ground_truth)
            
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            bbox_match = re.search(bbox_pattern, content_answer)
            if bbox_match:
                bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                bbox = [1000*(c/input_width) if i%2==0 else 1000*(c/input_height) for i,c in enumerate(bbox)]
                if iou(bbox, ground_truth) > iou_threshold:
                    reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails
                
    return reward

def iou_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    image_grid_thw = kwargs['image_grid_thw']
    input_width=image_grid_thw[0][2].item()*14
    input_height=image_grid_thw[0][1].item()*14
    progress = kwargs['global_step'] / kwargs['max_steps']
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = single_iou_reward(content, sol, progress, image_grid_thw)
        rewards.append(reward)
        # with open(log_path, "a") as f:
        try:
            print(f"------------- {current_time} RefExp reward: {reward} with progress {progress}-------------\n")
            print(f"Image size: width {input_width} with height {input_height}")
            print(f"Content: {content}\n")
            print(f"Solution: {sol}\n")
        except:
            print("log writing error")
    return rewards