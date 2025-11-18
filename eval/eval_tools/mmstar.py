from omegaconf import OmegaConf
import argparse, json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
import re

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

label2index = {'A':0, 'B': 1, 'C': 2, 'D': 3}

def MMStar_eval(data, args):
    MMStar_score_l2 = {
        'coarse perception': {
            'image scene and topic': 0,
            'image style & quality': 0,
            'image emotion': 0
        },
        'fine-grained perception': {
            'object counting': 0,
            'recognition': 0,
            'localization': 0
        },
        'instance reasoning': {
            'single-instance reasoning': 0,
            'cross-instance attribute reasoning': 0,
            'cross-instance relation reasoning': 0
        },
        'logical reasoning': {
            'code & sequence reasoning': 0,
            'diagram reasoning': 0,
            'common reasoning': 0
        },
        'science & technology': {
            'biology & chemistry & physics': 0,
            'electronics & energy & mechanical eng.': 0,
            'geography & earth science & agriculture': 0
        },
        'math': {
            'geometry': 0,
            'numeric commonsense and calculation': 0,
            'statistical reasoning': 0
        },
    }
    MMStar_counter = deepcopy(MMStar_score_l2)
    lt = len(data)
    missed = 0
    item_to_correct = []
    for i in tqdm(range(len(data))):
        line = data[i]
        predict = str(line['prediction'])
        answers = str(line['answer'])
        category = str(line['category'])
        l2_category = str(line['l2_category'])
        MMStar_counter[category][l2_category] += 1

        answer = answers.lower().strip().replace('\n', ' ')
        predict = predict.lower().strip().replace('\n', ' ')
        prev_num = MMStar_score_l2[category][l2_category]
        # if ori_bench == 'MathVista' and answer not in ['a', 'b', 'c', 'd']:
        #     if answer in predict:
        #         MMStar_score_l2[category][l2_category] += 1
        # else:
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
            if content_match:
                student_answer = content_match.group(1).strip()
                if '\\boxed' in student_answer or '\\text' in student_answer:
                    student_answer = extract_answer_from_special_tokens_fromDistill(student_answer)
                MMStar_score_l2[category][l2_category] += 1 if student_answer == answer else 0
            elif "<answer>" in predict:
                student_answer = predict.split("<answer>")[-1].strip()
                MMStar_score_l2[category][l2_category] += 1 if student_answer[0] == answer else 0
            elif len(predict) == 1:
                MMStar_score_l2[category][l2_category] += 1 if answer == predict[0] else 0
            elif '\\boxed' in predict or '\\text' in predict:
                student_answer = extract_answer_from_special_tokens_fromDistill(predict)
                MMStar_score_l2[category][l2_category] += 1 if student_answer == answer else 0
            elif predict[0] == '(' and answer == predict[1]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:7] == 'option ' and answer == predict[7]:
                MMStar_score_l2[category][l2_category] += 1
            elif predict[0:14] == 'the answer is ' and answer == predict[14]:
                MMStar_score_l2[category][l2_category] += 1
            elif len(re.findall('<answer>[a-z]', predict)) > 0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('<answer>[a-z]', predict)[0][8]==answer else 0
            elif len(re.findall('<answer>\([a-z]\)', predict)) > 0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('<answer>\([a-z]\)', predict)[0][9]==answer else 0
            elif len(re.findall('the answer is [a-z]', predict)) > 0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the answer is [a-z]', predict)[0][-1]==answer else 0
            elif len(re.findall('the correct answer is:\s+[a-z]', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct answer is:\s+[a-z]', predict)[0][-1]==answer else 0
            elif len(re.findall('the correct answer is\s+[a-z]', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct answer is\s+[a-z]', predict)[0][-1]==answer else 0
            elif len(re.findall('the correct option is:\s+[a-z]', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct option is:\s+[a-z]', predict)[0][-1]==answer else 0
            elif len(re.findall('the correct option is\s+[a-z]', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct option is\s+[a-z]', predict)[0][-1]==answer else 0
            elif len(re.findall('the correct option is\s+\([a-z]\)', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct option is\s+\([a-z]\)', predict)[0][-2]==answer else 0
            elif len(re.findall('the correct option is:*\s+\([a-z]\)', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct option is:*\s+\([a-z]\)', predict)[0][-2]==answer else 0
            elif len(re.findall('the correct answer is:*\s+\([a-z]\)', predict))>0:
                MMStar_score_l2[category][l2_category] += 1 if re.findall('the correct answer is:*\s+\([a-z]\)', predict)[0][-2]==answer else 0
            else:
                print("missed item: predict: {}, GT: {}".format(predict, answer))
                missed += 1
        except Exception as e:
            print("missed item: predict: {}, GT: {}".format(predict, answer))
            missed += 1
            pass
        item_to_correct.append(bool(MMStar_score_l2[category][l2_category] - prev_num))

    with open(args.result+".eval_out", 'w') as wf:
        json.dump(item_to_correct, wf)

    MMStar_score = {}
    MMStar_score['final score'] = 0
    for k, v in MMStar_score_l2.items():
        MMStar_score[k] = 0
        for l2_k, l2_v in v.items():
            MMStar_score[f'{k}({l2_k})'] = float(l2_v) / \
                float(MMStar_counter[k][l2_k])
            MMStar_score[k] += l2_v
        MMStar_score['final score'] += MMStar_score[k]
        MMStar_score[k] = float(MMStar_score[k]) / 250.0
    MMStar_score['final score'] = float(MMStar_score['final score']) / 1500.0

    # score_pth = eval_file.replace('.xlsx', '_score.json')
    # dump(MMStar_score, score_pth)
    print('Score: ')
    print("format miss rate: {}".format(missed / 1500))
    for key, value in MMStar_score.items():
        print('{}:{}'.format(key, value))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--path', type=str, default='/data/MMStar/')
    args = parser.parse_args()
    
    ds_meta = load_dataset(args.path, 'val')
    model_output = json.load(open(args.result))
    assert len(model_output) == len(ds_meta['val'])
    data = []
    res = model_output
    print("reasoning mode proportion: {:.2f} grounded and {:.2f} textual".format(len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<grounding>')]) / len(res), len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<text>')]) / len(res)))
    model_output = sorted(model_output, key=lambda item:int(item['item_id'].split('_')[1]))
    for i in range(len(model_output)):
        item = ds_meta['val'][i]
        del(item['image'])
        q = model_output[i]['question'].replace('<|image_pad|>', '')
        r = model_output[i]['response'].replace('<|image_pad|>', '')
        assert model_output[i]['label'] == item['answer']
        if r.startswith(q):
            item['prediction'] = r[len(q):].replace('<|im_end|>', '')
        else:
            item['prediction'] = r.replace('<|im_end|>', '')
        data.append(item)

    MMStar_eval(data, args)

if __name__=='__main__':
    main()