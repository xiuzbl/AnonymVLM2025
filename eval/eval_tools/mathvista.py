from omegaconf import OmegaConf
import argparse, json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
import re
from text_to_num import text2num

from math_verify import parse, verify
from eval.math_eval_utils import extract_answer, math_equal, get_multiple_choice_answer, mmlu_pro_extract_answer

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

def mathvista_think_process_results(doc):

    option_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def normalize(s):
        if isinstance(s, str):
            return '$'+s+'$'
        else:
            return s
    
    def pre_process(s, choices):
        if choices and s in choices:
            return chr(65+choices.index(s))
        return s
    
    def option_value_judge(gt_new, pred_new, options, pred):
        if (not pred_new in option_letter) and (gt_new in option_letter):
            if pred_new in options and options.index(pred_new) == ord(gt_new)-ord('A'):
                return True
            else:
                alpha_pred = mmlu_pro_extract_answer(pred)
                return alpha_pred == gt_new

        if (pred_new in option_letter) and (not gt_new in option_letter):
            if gt_new in options and options.index(gt_new) == ord(pred_new)-ord('A'):
                return True
        
        return False

    prediction = doc['prediction'].strip()

    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }

    result = {
        "question_id": doc["pid"],
        "query": doc["query"],
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "extraction": "",
        "prediction": prediction,
        "true_false": False,
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "precision": doc["precision"] if "precision" in doc else 0,
        "metadata": doc["metadata"],
    }

    pre_extrct_ans = extract_answer(prediction)
    if verify(parse(normalize(problem["answer"])), parse(normalize(pre_extrct_ans))):
        result["true_false"], result["extraction"] = True, pre_extrct_ans
        return {
            "mathvista_think_eval_score": result,
            "submission": result,
        }

    gt_new = pre_process(problem["answer"], problem["choices"])
    pred_new = get_multiple_choice_answer(prediction) if gt_new in option_letter else pre_extrct_ans
    if math_equal(gt_new, pred_new):
        result["true_false"], result["extraction"] = True, pred_new
        return {
            "mathvista_think_eval_score": result,
            "submission": result,
        }
    else:
        gt_parse = parse(normalize(gt_new))
        pred_parse = parse(normalize(pred_new))
        if verify(gt_parse, pred_parse):
            result["true_false"], result["extraction"] = True, pred_parse
            return {
                "mathvista_think_eval_score": result,
                "submission": result,
            }
        
    # 处理gt和预测答案一个选项一个数值的情况
    is_correct = option_value_judge(gt_new, pred_new, problem["choices"], prediction)
    result["true_false"], result["extraction"] = is_correct, pred_new

    return {
        "mathvista_think_eval_score": result,
        "submission": result,
    }

def CLEVR_eval(data, args):
    class2res = defaultdict(list)
    lt = len(data)
    right = 0
    missed = 0
    mse = []
    result_by_category = {}
    result_by_format = {"hit": [], "miss": []}
    result_by_type = {'free_form':[], 'multi_choice':[]}
    item_to_correct = []
    for i in tqdm(range(len(data))):
        line = data[i]
        predict = str(line['prediction'])
        # answers = str(line['label'])
        # category = line['category']

        # answer = answers.lower().strip().replace('\n', ' ')
        # predict = predict.lower().strip().replace('\n', ' ')
        flag = False
        question_type = line['question_type']
        correct_flag = False
        try:
            content_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
            if args.extract_from_answer:
                student_answer = content_match.group(1).strip() if content_match else predict.strip()
            else:
                student_answer = predict.strip()
            # if '\\boxed' in student_answer: # or '\\text' in student_answer:
            #     student_answer = extract_answer_from_special_tokens_fromDistill(student_answer)
            line['prediction'] = student_answer
            line_result = mathvista_think_process_results(line)
            if line_result['submission']['true_false']:
                class2res[line['metadata']['category']].append(1)
                correct_flag = True
                right += 1
            else:
                class2res[line['metadata']['category']].append(0)
            if content_match:
                result_by_format["hit"].append(1 if line_result['submission']['true_false'] else 0)
            else:
                result_by_format["miss"].append(1 if line_result['submission']['true_false'] else 0)
            result_by_type[question_type].append(1 if line_result['submission']['true_false'] else 0)
            # if content_match:
        except Exception as e:
            print("missed item: {}".format(predict))
            missed += 1
            pass
        item_to_correct.append(correct_flag)
    
    # save the item_to_correct json
    with open(args.result+".eval_out", 'w') as wf:
        json.dump(item_to_correct, wf)


    # score_pth = eval_file.replace('.xlsx', '_score.json')
    # dump(MMStar_score, score_pth)
    print('Evaluation Error Rate: {}'.format(missed / lt))
    print('ALL Score: {}'.format(right/lt))
    for key, value in class2res.items():
        print("Category {} Score: {}".format(key, sum(value)/len(value)))
    # print("MSE: {}".format(sum(mse)/(len(mse)+0.00001)))
    print("=================Performance of Different Formats============")
    print("There are {} percent data that do not include <answer> </answer>".format(100*len(result_by_format['miss'])/lt))
    print("Acc on the format-hit part: {}".format(sum(result_by_format['hit'])/(len(result_by_format['hit'])+0.0001)))
    print("Acc on the format-miss part: {}".format(sum(result_by_format['miss'])/(len(result_by_format['miss'])+0.0001)))
    print("=================Performance of Different Question Types============")
    for k,v in result_by_type.items():
        print("Acc on {} questions: {}".format(k, sum(v)/len(v)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--path', type=str, default='/data/MathVista/')
    parser.add_argument("--lmms_eval", action="store_true")
    parser.add_argument("--extract_from_answer", action="store_true")
    args = parser.parse_args()
    
    model_output = json.load(open(args.result))
    ds_meta = load_dataset(args.path)['testmini']
    data = []
    res = model_output
    print("reasoning mode proportion: {:.2f} grounded and {:.2f} textual".format(len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<grounding>')]) / len(res), len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<text>')]) / len(res)))
    model_output = sorted(model_output, key=lambda item:int(item['item_id'].split('_')[-1]))
    if args.lmms_eval:
        for i in range(len(model_output['logs'])):
            item = model_output['logs'][i]
            item['prediction'] = item['resps'][0][0]
            data.append(item)
    else:
        for i in range(len(model_output)):
            item = ds_meta[i]
            q = model_output[i]['question'].replace('<|image_pad|>', '')
            r = model_output[i]['response'].replace('<|image_pad|>', '')
            if r.startswith(q):
                item['prediction'] = r[len(q):].replace('<|im_end|>', '')
            else:
                item['prediction'] = r.replace('<|im_end|>', '')
            del(item['decoded_image'])
            data.append(item)

    CLEVR_eval(data, args)

if __name__=='__main__':
    main()