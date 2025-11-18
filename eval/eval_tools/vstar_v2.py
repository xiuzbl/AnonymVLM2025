from omegaconf import OmegaConf
import argparse, json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
import re

label2index = {'A':0, 'B': 1, 'C': 2, 'D': 3}
option_letters = ['A', 'B', 'C', 'D']

def extract_answer_from_special_tokens_fromDistill(content):
    # 提取<answer>标签内的内容
    # answer_content = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    # if not answer_content:
    #     return None
    # answer_text = answer_content.group(1)
    
    # 查找所有被\boxed{}或\text{}包裹的数值
    matches = re.findall(r'\\(?:boxed)\{([^}]*)\}', content)
    
    # 返回第一个匹配结果，如果没有则返回None
    return matches[0] if matches else None

def embspatial_eval(data, meta, args):
    class2res = defaultdict(list)
    lt = len(data)
    missed = 0
    item_to_correct = []
    for i in tqdm(range(len(data))):
        line = data[i]
        predict = str(line['prediction'])
        answers = meta[i]['label']
        category = meta[i]['category']

        answer = answers.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ')
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
            if content_match:
                student_answer = content_match.group(1).strip()
                if '\\boxed' in student_answer or '\\text' in student_answer:
                    student_answer = extract_answer_from_special_tokens_fromDistill(student_answer)
                elif student_answer[0:13] == 'final answer:':
                    student_answer = student_answer[13:].strip()
                class2res[category].append(1 if student_answer == answer else 0)
            elif "<answer>" in predict:
                student_answer = predict.split("<answer>")[-1].strip()
                class2res[category].append(1 if answer == student_answer[0] else 0)
            elif len(predict) == 1:
                class2res[category].append(1 if answer == predict[0] else 0)
            elif '\\boxed' in predict or '\\text' in predict:
                student_answer = extract_answer_from_special_tokens_fromDistill(predict)
                class2res[category].append(1 if student_answer == answer else 0)
            elif predict[0] == '(' and answer == predict[1]:
                class2res[category].append(1)
            elif predict[0:7] == 'option ' and answer == predict[7]:
                class2res[category].append(1)
            elif predict[0:14] == 'the answer is ' and answer == predict[14]:
                class2res[category].append(1)
            elif predict[0:14] == 'final answer:' and answer == predict[14]:
                class2res[category].append(1)
            elif len(re.findall('<answer>[a-z]', predict)) > 0:
                class2res[category].append(1 if re.findall('<answer>[a-z]', predict)[0][8]==answer else 0)
            elif len(re.findall('<answer>\([a-z]\)', predict)) > 0:
                class2res[category].append(1 if re.findall('<answer>\([a-z]\)', predict)[0][9]==answer else 0)
            elif len(re.findall('the answer is [a-z]', predict)) > 0:
                class2res[category].append(1 if re.findall('the answer is [a-z]', predict)[0][-1]==answer else 0)
            elif len(re.findall('<answer> [a-z]', predict)) > 0:
                class2res[category].append(1 if re.findall('<answer> [a-z]', predict)[0][9]==answer else 0)
            elif len(re.findall('<answer> \([a-z]\)', predict)) > 0:
                class2res[category].append(1 if re.findall('<answer> \([a-z]\)', predict)[0][10]==answer else 0)
            elif len(re.findall('the correct answer is:*\s+[a-z]', predict))>0:
                class2res[category].append(1 if re.findall('the correct answer is:*\s+[a-z]', predict)[0][-1]==answer else 0)
            elif len(re.findall('the correct option is:*\s+[a-z]', predict))>0:
                class2res[category].append(1 if re.findall('the correct option is:*\s+[a-z]', predict)[0][-1]==answer else 0)
            else:
                print("missed item: predict: {}, GT: {}".format(predict, answer))
                missed += 1
                class2res[category].append(0)
        except Exception as e:
            print("missed item: predict: {}, GT: {}".format(predict, answer))
            class2res[category].append(0)
            pass
        item_to_correct.append(bool(class2res[category][-1]))

    with open(args.result+".eval_out", 'w') as wf:
        json.dump(item_to_correct, wf)
    
    clevr_score = {}
    clevr_score['final score'] = 0
    total_sum = 0
    for k, v in class2res.items():
        clevr_score[k] = sum(v) / len(v)
        total_sum += len(v)
        clevr_score['final score'] += sum(v)
    clevr_score['final score'] = clevr_score['final score'] / total_sum

    # score_pth = eval_file.replace('.xlsx', '_score.json')
    # dump(MMStar_score, score_pth)
    print('Score: ')
    print("Format missed: {}".format(missed/total_sum))
    for key, value in clevr_score.items():
        print('{}:{}'.format(key, value))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--path', type=str, default='/data/vstar_bench/test_questions.jsonl')
    args = parser.parse_args()
    
    ds_meta = [json.loads(line) for line in open(args.path)]
    model_output = json.load(open(args.result))
    assert len(model_output) == len(ds_meta)
    data = []
    res = model_output
    print("reasoning mode proportion: {:.2f} grounded and {:.2f} textual".format(len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<grounding>')]) / len(res), len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<text>')]) / len(res)))
    model_output = sorted(model_output, key=lambda item:int(item['item_id'].split('_')[1]))
    for i in range(len(model_output)):
        item = {}
        q = model_output[i]['question'].replace('<|image_pad|>', '')
        r = model_output[i]['response'].replace('<|image_pad|>', '')
        if r.startswith(q):
            item['prediction'] = r[len(q):].replace('<|im_end|>', '')
        else:
            item['prediction'] = r.replace('<|im_end|>', '')
        data.append(item)

    embspatial_eval(data, ds_meta, args)

if __name__=='__main__':
    main()