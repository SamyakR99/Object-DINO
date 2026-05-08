#!/usr/bin/env python3
"""
Simple script to evaluate MME results.
Usage: python simple_eval.py mme_results_alpha_0.4_tokens_64.json
"""

import os
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

eval_type_dict = {
    "Perception": ["existence", "count"],
    "Cognition": ["position", "color"]
}

def divide_chunks(l, n=2):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def parse_pred_ans(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label

def compute_metric(gts, preds):
    assert len(gts) == len(preds)
    
    label_map = {"yes": 1, "no": 0, "other": -1}
    gts = [label_map[x] for x in gts]
    preds = [label_map[x] for x in preds]
    
    acc = accuracy_score(gts, preds)
    
    clean_gts = []
    clean_preds = []
    other_num = 0
    for gt, pred in zip(gts, preds):
        if pred == -1:
            other_num += 1
            continue
        clean_gts.append(gt)
        clean_preds.append(pred)
    
    conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
    precision = precision_score(clean_gts, clean_preds, average='binary')
    recall = recall_score(clean_gts, clean_preds, average='binary')
    tp, fn = conf_mat[0]
    fp, tn = conf_mat[1]
    
    metric_dict = {
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "precision": precision, "recall": recall,
        "other_num": other_num, "acc": acc,
    }
    return metric_dict

def clean_question(question):
    """Remove instruction suffixes from question."""
    question = question.strip()
    question = question.replace("\nAnswer the question using a single word or phrase.", "")
    question = question.replace("Answer the question using a single word or phrase.", "")
    question = question.replace("Please answer yes or no.", "")
    question = question.replace("  Please answer yes or no.", "")
    return question.strip()

def load_ground_truth(data_path):
    """Load ground truth from MME benchmark."""
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        
        if os.path.exists(os.path.join(category_dir, 'images')):
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            qa_path = category_dir
        
        if not os.path.isdir(qa_path):
            continue
        
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            with open(os.path.join(qa_path, file), 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        question, answer = parts[0], parts[1]
                        GT[(category, file, question)] = answer
    return GT

def evaluate_json(json_file, data_path, method='greedy'):
    """Evaluate results directly from JSON file."""
    
    print(f"Loading ground truth from {data_path}...")
    GT = load_ground_truth(data_path)
    print(f"Loaded {len(GT)} GT entries")
    
    print(f"\nLoading predictions from {json_file}...")
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")
    
    # Organize by category
    category_data = {}
    matched = 0
    unmatched = 0
    
    for pred in predictions:
        img_path = pred['original_filename']
        category = img_path.split('/')[0] if '/' in img_path else 'unknown'
        
        if category not in eval_type_dict["Perception"] + eval_type_dict["Cognition"]:
            continue
        
        if category not in category_data:
            category_data[category] = []
        
        # Get filename and question
        filename = os.path.basename(img_path)
        file_txt = os.path.splitext(filename)[0] + '.txt'
        
        base_question = clean_question(pred['question'])
        gt_question = base_question + " Please answer yes or no."
        
        # Check if GT exists
        if (category, file_txt, gt_question) not in GT:
            gt_question = base_question + "  Please answer yes or no."  # Try double space
        
        if (category, file_txt, gt_question) in GT:
            gt_ans = GT[(category, file_txt, gt_question)]
            pred_ans = pred[method]
            category_data[category].append((gt_ans, pred_ans))
            matched += 1
        else:
            unmatched += 1
    
    print(f"\nMatched: {matched}, Unmatched: {unmatched}")
    
    # Evaluate by category
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({method.upper()})")
    print(f"{'='*60}")
    
    total_scores = {}
    for eval_type, task_list in eval_type_dict.items():
        print(f"\n{eval_type}:")
        print("-" * 40)
        
        type_score = 0
        for task in task_list:
            if task not in category_data:
                print(f"  {task}: No data")
                continue
            
            pairs = category_data[task]
            chunks = list(divide_chunks(pairs))
            
            gts = []
            preds = []
            acc_plus_correct = 0
            
            for chunk in chunks:
                chunk_correct = 0
                for gt, pred in chunk:
                    gt = gt.lower()
                    pred = parse_pred_ans(pred.lower())
                    gts.append(gt)
                    preds.append(pred)
                    if gt == pred:
                        chunk_correct += 1
                if chunk_correct == 2:
                    acc_plus_correct += 1
            
            metrics = compute_metric(gts, preds)
            acc_plus = acc_plus_correct / len(chunks)
            
            task_score = metrics['acc'] * 100 + acc_plus * 100
            type_score += task_score
            
            print(f"  {task}:")
            print(f"    Score: {task_score:.2f}")
            print(f"    Accuracy: {metrics['acc']*100:.2f}%")
            print(f"    Acc+: {acc_plus*100:.2f}%")
            print(f"    Precision: {metrics['precision']*100:.2f}%")
            print(f"    Recall: {metrics['recall']*100:.2f}%")
        
        total_scores[eval_type] = type_score
        print(f"\n  {eval_type} Total Score: {type_score:.2f}")
    
    print(f"\n{'='*60}")
    overall = sum(total_scores.values())
    print(f"OVERALL SCORE: {overall:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_eval.py <json_file> [method] [data_path]")
        print("  method: 'greedy' or 'sampling' (default: greedy)")
        print("  data_path: path to MME dataset (default: /scratch/bcyh/dataset/MME_Benchmark_release_version/)")
        sys.exit(1)
    
    json_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'greedy'
    data_path = sys.argv[3] if len(sys.argv) > 3 else '/scratch/bcyh/dataset/MME_Benchmark_release_version/'
    
    evaluate_json(json_file, data_path, method)