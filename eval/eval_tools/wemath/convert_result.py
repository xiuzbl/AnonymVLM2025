import argparse
import json
from datasets import load_dataset
import re

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

def reduce_to_option(item, predict):
    if predict not in ["a", 'b', 'c', 'd', 'e', 'A', 'B', 'C', 'D', 'E']:
        options = [opt.strip()[3:] for opt in item['option'].split(";")]
        if predict in options:
            predict = "ABCDEFGHIJK"[options.index(predict)]
        return predict
    else:
        return predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--grd_result", type=str, default=None)
    parser.add_argument("--path", type=str, default="/data/We-Math/")
    parser.add_argument("--reduce_to_option", action="store_true")
    args = parser.parse_args()

    model_output = json.load(open(args.result))
    if args.grd_result is not None:
        grd_result = json.load(open(args.grd_result))
    else:
        grd_result = []
    ds_meta = load_dataset(args.path)['testmini']
    if args.output is None:
        args.output = args.result + ".eval"
    data = []
    res = model_output
    print("reasoning mode proportion: {:.2f} grounded and {:.2f} textual".format(len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<grounding>')]) / len(res), len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<text>')]) / len(res)))
    model_output = sorted(model_output, key=lambda item:int(item['item_id'].split('_')[-1]))
    grd_result = sorted(grd_result, key=lambda item:int(item['item_id'].split('_')[-1]))
    c = 0
    for i in range(len(model_output)):
        item = ds_meta[i]
        q = model_output[i]['question'].replace('<|image_pad|>', '')
        r = model_output[i]['response'].replace('<|image_pad|>', '')
        if r.startswith(q):
            item['response'] = r[len(q):].replace('<|im_end|>', '')
        else:
            item['response'] = r.replace('<|im_end|>', '')
        predict = item['response']
        content_match = re.search(r'<answer>(.*?)</answer>', predict, re.DOTALL)
        if content_match:
            student_answer = content_match.group(1).strip()
        else:
            c += 1
            if grd_result:
                content_match = re.search(r'<answer>(.*?)</answer>', grd_result[i]['response'], re.DOTALL)
                if content_match:
                    student_answer = content_match.group(1).strip()
                else:
                    student_answer = grd_result[i]['response']
            else:
                student_answer = predict
        if "\\boxed" in student_answer:
            student_answer = extract_answer_from_special_tokens_fromDistill(student_answer)
        if student_answer.endswith('.'):
            student_answer = student_answer.strip(".")
        if args.reduce_to_option:
            student_answer = reduce_to_option(item, student_answer)
        item['response'] = "<Answer>: {}".format(student_answer)
        del(item['image_path'])
        data.append(item)
    print("there are {} percent data without valida answer tag".format(100*c/len(model_output)))
    
    with open(args.output, 'w') as wf:
        json.dump(data, wf)

if __name__=="__main__":
    main()