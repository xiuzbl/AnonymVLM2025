from omegaconf import OmegaConf
import argparse, json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
import re
from text_to_num import text2num

def extract_answer_from_special_tokens_fromDistill(content):
    # 提取<answer>标签内的内容
    # answer_content = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    # if not answer_content:
    #     return None
    # answer_text = answer_content.group(1)
    
    # 查找所有被\boxed{}或\text{}包裹的数值
    # matches = re.findall(r'\\(?:boxed|text)\{([^}]*)\}', content)
    matches = re.findall(r'\\(?:boxed)\{([^}]*)\}', content)
    
    # 返回第一个匹配结果，如果没有则返回None
    return matches[0] if matches else None

def CLEVR_eval(data, args):
    class2res = defaultdict(list)
    lt = len(data)
    right = 0
    missed = 0
    mse = []
    result_by_category = {}
    item_to_correct = []
    for i in tqdm(range(len(data))):
        line = data[i]
        predict = str(line['prediction'])
        answers = str(line['label'])
        category = line['category']

        answer = answers.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ').rstrip('.')
        flag = False
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else predict.strip()
            content_match = re.search(r'answer>(.*?)</answer>', predict, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else predict.strip()
            if '\\boxed' in student_answer: # or '\\text' in student_answer:
                student_answer = extract_answer_from_special_tokens_fromDistill(student_answer)
            if student_answer.startswith('final answer:'):
                student_answer = student_answer[13:].strip()
            if "<answer>" in student_answer:
                student_answer = student_answer.split('<answer>')[-1].strip()
            if student_answer.startswith('yes') or "answer: yes" in student_answer or student_answer.endswith("yes") or "the answer is yes" in student_answer:
                flag = answers == "yes"
                right += 1 if answers == "yes" else 0
            elif student_answer.startswith('no') or "answer: no" in student_answer or student_answer.endswith("no") or "the answer is no" in student_answer:
                flag = answers == "no"
                right += 1 if answers == "no" else 0
            else:
                # predict_int = text2num(student_answer, 'en')
                print("missed item1: {}".format(predict))
                print("gt answer: {}".format(answers))
                print("=="*10)
                missed += 1
            # if content_match:
        except Exception as e:
            print("missed item: {}".format(predict))
            missed += 1
            pass
        if category not in result_by_category:
            result_by_category[category] = {"correct": 0, "all": 0}
        item_to_correct.append(flag)
        result_by_category[category]['all'] += 1
        result_by_category[category]['correct'] += (1 if flag else 0)


    with open(args.result+".eval_out", 'w') as wf:
        json.dump(item_to_correct, wf)
    # score_pth = eval_file.replace('.xlsx', '_score.json')
    # dump(MMStar_score, score_pth)
    print('Format Miss Rate: {}'.format(missed / lt))
    print('ALL Score: {}'.format(right/lt))
    for key, value in result_by_category.items():
        print("{} Score: {}".format(key, value['correct']/value['all']))
    # print("MSE: {}".format(sum(mse)/(len(mse)+0.00001)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--path', type=str, default='/data/POPE/')
    args = parser.parse_args()
    
    model_output = json.load(open(args.result))
    ds_meta = load_dataset(args.path)['test']
    data = []
    res = model_output
    print("reasoning mode proportion: {:.2f} grounded and {:.2f} textual".format(len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<grounding>')]) / len(res), len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<text>')]) / len(res)))
    model_output = sorted(model_output, key=lambda item:int(item['item_id'].split('_')[-1]))
    for i in range(len(model_output)):
        item = ds_meta[i]
        q = model_output[i]['question'].replace('<|image_pad|>', '')
        r = model_output[i]['response'].replace('<|image_pad|>', '')
        if r.startswith(q):
            item['prediction'] = r[len(q):].replace('<|im_end|>', '')
        else:
            item['prediction'] = r.replace('<|im_end|>', '')
        item['label'] = model_output[i]['label']
        del(item['image'])
        data.append(item)

    CLEVR_eval(data, args)

if __name__=='__main__':
    main()