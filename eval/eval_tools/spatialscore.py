from omegaconf import OmegaConf
import argparse, json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
import re
import os
from eval.eval_tools.eval_utils import image_to_base64_data_uri, extract_number, extract_yes_no, extract_option, extract_numeric_with_unit, load_image, mean_relative_accuracy

def extract_answer_from_special_tokens_fromDistill(content):
    # 提取<answer>标签内的内容
    # answer_content = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    # if not answer_content:
    #     return None
    # answer_text = answer_content.group(1)
    
    # 查找所有被\boxed{}或\text{}包裹的数值
    matches = re.findall(r'\\(?:boxed)\s*\{([^}]*)\}', content, re.DOTALL)
    
    # 返回第一个匹配结果，如果没有则返回None
    if matches:
        return matches[0]
    else:
        matches = re.findall(r'\\(?:boxed)\s*\(([^}]*)\)', content, re.DOTALL)
        return matches[0] if matches else None

label2index = {'A':0, 'B': 1, 'C': 2, 'D': 3}

def SpatialScoe_eval(data, output_dir, args):
    lt = len(data)
    missed = 0
    all_results, source_results, category_results = [], {}, {}
    total_correct, total_samples = 0, 0
    item_to_correct = []
    for i in tqdm(range(len(data))):
        item = data[i]
        # Determine whether response is correct based on question type
        is_correct = False
        question_type = item.get('question_type', '')
        ground_truth = item.get('answer', '')
        results = str(item['prediction'])
        content_match = re.search(r'<answer>(.*?)</answer>', results, re.DOTALL)
        if content_match:
            results = content_match.group(1).strip()
        elif "<answer>" in results:
            results = results.split("<answer>")[-1].strip()
        else:
            item_to_correct.append(False)
            continue
        before_processed = results
        if '\\boxed' in results:
            results = extract_answer_from_special_tokens_fromDistill(results)

        if question_type.lower() == 'multi-choice':
            # Extract options for multiple choice questions
            pred_answer = extract_option(results)
            gt_answer = extract_option(ground_truth)
            is_correct = pred_answer.upper() == gt_answer.upper()
        
        elif question_type.lower() == 'judgment':
            # Extract yes/no answers for judgment questions
            try:
                pred_answer = extract_yes_no(results)
            except:
                print(before_processed)
            gt_answer = extract_yes_no(ground_truth)
            is_correct = pred_answer.lower() == gt_answer.lower()
        
        else:  # Open-ended questions
            if any(unit in ground_truth.lower() for unit in ['meter', 'meters', 'm', 'cm', 'centimeter', 'centimeters', 'km', 'kilometer', 'kilometers', 'inch', 'inches', 'ft', 'foot', 'feet']):
                # Extract numerical value with unit for open-ended questions with units
                is_correct = extract_numeric_with_unit(results, ground_truth)['is_correct']

            elif item.get('source') == 'RealWorldQA':
                is_correct = results.lower() == ground_truth.lower()  # Exact match for RealWorldQA
            
            else:
                # For other open-ended questions, extract numbers
                try:
                    pred_value = float(extract_number(results))
                except:
                    pred_value = 0.0
                gt_value = float(extract_number(ground_truth))
                
                # Check if both values are extracted successfully
                if pred_value is not None and gt_value is not None:
                    if item.get('source') == 'VSI-Bench_8':
                        is_correct = False
                        if pred_value == 0:
                            score = 1.0 if gt_value == 0.0 else 0.0
                        else:
                            # from utils.util import mean_relative_accuracy
                            score = mean_relative_accuracy(pred_value, gt_value, start=0.5, end=0.95, interval=0.05)
                            # ratio = max(pred_value / gt_value, gt_value / pred_value)
                            # is_correct = ratio <= 2.0  # Allow for delta = 2 tolerance
                    else:
                        is_correct = pred_value == gt_value # Exact match for other datasets (SpatialBench, RealWorldQA)


        # Update counters
        if is_correct == True:
            total_correct += 1
            score = 1.0
        elif is_correct == False:
            score = 0.0
        
        item_to_correct.append(is_correct)

        total_samples += 1
        
        # Create result entry
        result_entry = {
            "id": item.get('id', i), "category": item.get('category', 'unknown'), "subcategory": item.get('subcategory', 'unknown'),
            "input_modality": item.get('input_modality', 'image'), "question_type": question_type, "source": item.get('source', 'unknown'), 
            "question": item.get('question', ''), "gt_answer": ground_truth, "pred_answer": results, "img_paths": item.get('img_paths', []), "is_correct": is_correct, "score": score
        }
        
        # Add to all results
        all_results.append(result_entry)
        
        # Group by source
        source = item.get('source', 'unknown')
        if source not in source_results:
            source_results[source] = []
        source_results[source].append(result_entry)

        # Group by category
        category = item.get('category', 'unknown')
        if category not in category_results:
            category_results[category] = []
        category_results[category].append(result_entry)
          

    with open(args.result+".eval_out", 'w') as wf:
        json.dump(item_to_correct, wf)
    # Save results grouped by source
    for source, results in source_results.items():
        source_dir = os.path.join(output_dir, "by_source")
        os.makedirs(source_dir, exist_ok=True)
        
        with open(os.path.join(source_dir, f"{source}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate and save source-level accuracy using score instead of is_correct
        source_score_sum = sum(r.get('score', 0.0) for r in results)
        source_correct = int(source_score_sum)  # Floor the score as requested
        source_total = len(results)
        source_accuracy = (source_score_sum / source_total) * 100 if source_total > 0 else 0
        
        with open(os.path.join(source_dir, f"{source}_summary.json"), 'w') as f:
            summary = {"source": source, "accuracy": source_accuracy, "correct": source_correct, "total": source_total, "score_sum": source_score_sum}
            json.dump(summary, f, indent=2)
        
        print(f"Source: {source} - Accuracy: {source_accuracy:.2f}% ({source_correct}/{source_total}, score: {source_score_sum:.2f})")
    
    # Save results grouped by category
    for category, results in category_results.items():
        category_dir = os.path.join(output_dir, "by_category")
        os.makedirs(category_dir, exist_ok=True)
        
        with open(os.path.join(category_dir, f"{category}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate and save category-level accuracy using score instead of is_correct
        category_score_sum = sum(r.get('score', 0.0) for r in results)
        category_correct = int(category_score_sum)  # Floor the score as requested
        category_total = len(results)
        category_accuracy = (category_score_sum / category_total) * 100 if category_total > 0 else 0
        
        with open(os.path.join(category_dir, f"{category}_summary.json"), 'w') as f:
            summary = {"category": category, "accuracy": category_accuracy, "correct": category_correct, "total": category_total, "score_sum": category_score_sum}
            json.dump(summary, f, indent=2)
        
        print(f"Category: {category} - Accuracy: {category_accuracy:.2f}% ({category_correct}/{category_total}, score: {category_score_sum:.2f})")

    # Calculate and save overall accuracy using score instead of counting is_correct=True
    total_score_sum = sum(r.get('score', 0.0) for r in all_results)
    total_correct = int(total_score_sum)  # Floor the score as requested
    overall_accuracy = (total_score_sum / total_samples) * 100 if total_samples > 0 else 0

    with open(os.path.join(output_dir, "overall_summary.json"), 'w') as f:
        summary = {"accuracy": overall_accuracy, "correct": total_correct, "total": total_samples, "score_sum": total_score_sum}
        json.dump(summary, f, indent=2)

    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples}, score: {total_score_sum:.2f})")
    print(f"All results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--meta_path', type=str, default='/data/SpatialScore/dataset/SpatialScore-Hard.json')
    args = parser.parse_args()
    
    ds_meta = json.load(open(args.meta_path))
    model_output = json.load(open(args.result))
    assert len(model_output) == len(ds_meta)
    data = []
    res = model_output
    print("reasoning mode proportion: {:.2f} grounded and {:.2f} textual".format(len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<grounding>')]) / len(res), len([item for item in res if item['response'].split('assistant\n')[-1].startswith('<text>')]) / len(res)))
    model_output = sorted(model_output, key=lambda item:int(item['item_id'].split('_')[1]))
    for i in range(len(model_output)):
        item = ds_meta[i]
        # del(item['image'])
        q = model_output[i]['question'].replace('<|image_pad|>', '')
        r = model_output[i]['response'].replace('<|image_pad|>', '')
        # assert model_output[i]['answer'] == item['answer']
        if r.startswith(q):
            item['prediction'] = r[len(q):].replace('<|im_end|>', '')
        else:
            item['prediction'] = r.replace('<|im_end|>', '')
        data.append(item)

    output_dir = os.path.join(os.path.dirname(args.result), "spatial_score_eval")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    SpatialScoe_eval(data, output_dir, args)

if __name__=='__main__':
    main()