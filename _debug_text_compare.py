import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

import torch
from bert_score import score as score_bert
from omegaconf import OmegaConf
from nlgmetricverse import NLGMetricverse, load_metric


def load_our_text_prediction_humanml():
    """Load our direct text predictions from .npy result files for humanml."""
    pred_dir = "/scratch/bfyo/tcheng1/exp_test_finetuned_humanml_6imu_1000frame/viz_test_generate_number_original"
    gt_folder = "/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence/motion_data/test"
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    
    pred_texts = []
    gt_texts = []
    for file in tqdm(files, desc="Loading predictions and ground truth"):
        data = np.load(file, allow_pickle=True).item()
        sample_idx = data['sample_idx'][0]
        pred_text = data['pred']['description']
        
        sample = pickle.load(open(f"{gt_folder}/{sample_idx}.pkl", "rb"))
        gt_text = sample['texts']  # list of text
        
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)
    
    return pred_texts, gt_texts


def load_our_motion_motiongpt3_humanml():
    """Load our motion + MotionGPT3 text predictions for humanml."""
    pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_text_pred_from_scratch"
    gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_humanml_6imu_60frame/viz_test_generate_number"
    test_dataset_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence/motion_data/test'
    
    number_of_samples = len(os.listdir(pred_texts_dir))
    
    pred_texts = []
    for i in tqdm(range(number_of_samples), desc="Loading predictions"):
        with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
            pred_text = f.read().strip()
        pred_texts.append(pred_text)
    
    gt_texts = []
    for i in tqdm(range(number_of_samples), desc="Loading ground truth"):
        gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
        test_sample_id = gt_info['sample_idx'][0]
        with open(os.path.join(test_dataset_dir, f"{test_sample_id}.pkl"), 'rb') as f:
            data = pickle.load(f)
        gt_text = data['texts']
        gt_texts.append(gt_text)
    
    return pred_texts, gt_texts


def save_text_comparison(pred_texts, gt_texts, output_path):
    """Save text comparison to a .txt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i, (pred, gt) in enumerate(zip(pred_texts, gt_texts)):
            f.write(f"Sample {i:04d}:\n")
            f.write(f"--> Pred:\n")
            f.write(f"    {pred}\n")
            
            # Handle both string and list of strings for ground truth
            if isinstance(gt, list):
                for j, gt_item in enumerate(gt):
                    f.write(f"--> GT {j+1:02d}:\n")
                    f.write(f"    {gt_item}\n")
            else:
                f.write(f"--> GT 01:\n")
                f.write(f"    {gt}\n")
            
            f.write("\n")
    
    print(f"Comparison saved to: {output_path}")


def load_and_unify_predictions_humanml():
    """Load and unify both prediction methods by matching sample_idx."""
    # Load e2e predictions with sample_idx
    pred_dir_e2e = "/scratch/bfyo/tcheng1/exp_test_finetuned_humanml_6imu_1000frame/viz_test_generate_number_original"
    files_e2e = sorted(glob.glob(os.path.join(pred_dir_e2e, "*.npy")))
    
    e2e_data = {}
    for file in tqdm(files_e2e, desc="Loading e2e predictions"):
        data = np.load(file, allow_pickle=True).item()
        sample_idx = data['sample_idx'][0]
        pred_text = data['pred']['description']
        e2e_data[sample_idx] = pred_text
    
    # Load mgpt3 predictions with sample_idx
    pred_texts_dir_mgpt3 = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_text_pred_from_scratch"
    gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_humanml_6imu_60frame/viz_test_generate_number"
    test_dataset_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence/motion_data/test'
    
    number_of_samples = len(os.listdir(pred_texts_dir_mgpt3))
    
    mgpt3_data = {}
    gt_data = {}
    for i in tqdm(range(number_of_samples), desc="Loading mgpt3 predictions and GT"):
        # Load mgpt3 prediction
        with open(os.path.join(pred_texts_dir_mgpt3, f"id_{i}_step_0.txt"), 'r') as f:
            pred_text = f.read().strip()
        
        # Get sample_idx from gt_texts_dir
        gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
        sample_idx = gt_info['sample_idx'][0]
        
        # Load ground truth
        with open(os.path.join(test_dataset_dir, f"{sample_idx}.pkl"), 'rb') as f:
            data = pickle.load(f)
        gt_text = data['texts']
        
        mgpt3_data[sample_idx] = pred_text
        gt_data[sample_idx] = gt_text
    
    return e2e_data, mgpt3_data, gt_data


def save_unified_comparison(e2e_data, mgpt3_data, gt_data, output_path):
    """Save unified comparison to a .txt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get common sample indices (sorted)
    common_samples = sorted(set(e2e_data.keys()) & set(mgpt3_data.keys()) & set(gt_data.keys()))
    
    with open(output_path, 'w') as f:
        for sample_idx in common_samples:
            f.write(f"Sample: {sample_idx}\n")
            f.write(f"--> Pred (ours-e2e):\n")
            f.write(f"    {e2e_data[sample_idx]}\n")
            f.write(f"--> Pred (ours-mgpt3):\n")
            f.write(f"    {mgpt3_data[sample_idx]}\n")
            
            # Handle both string and list of strings for ground truth
            gt = gt_data[sample_idx]
            if isinstance(gt, list):
                for j, gt_item in enumerate(gt):
                    f.write(f"--> GT {j+1:02d}:\n")
                    f.write(f"    {gt_item}\n")
            else:
                f.write(f"--> GT 01:\n")
                f.write(f"    {gt}\n")
            
            f.write("\n")
    
    print(f"Unified comparison saved to: {output_path}")
    print(f"Total matched samples: {len(common_samples)}")


def initialize_nlg_evaluator():
    """Initialize NLG evaluator for text metrics."""
    metrics = [
        load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
        load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
        load_metric("rouge"),
        load_metric("cider"),
    ]
    nlg_evaluator = NLGMetricverse(metrics)
    return nlg_evaluator


def compute_single_sample_metrics(pred_text, gt_text, nlg_evaluator, device='cuda'):
    """Compute metrics for a single sample.
    
    Args:
        pred_text: str, predicted text
        gt_text: str or list of str, ground truth text(s)
        nlg_evaluator: NLGMetricverse evaluator
        device: device for BERTScore computation
    
    Returns:
        dict of metric scores
    """
    if isinstance(gt_text, str):
        gt_text_list = [gt_text]
    else:
        gt_text_list = gt_text
    
    # Compute NLG metrics
    scores = nlg_evaluator(predictions=[pred_text], references=[gt_text_list])
    
    metrics = {}
    
    # Extract BLEU scores
    if 'bleu_1' in scores:
        metrics['BLEU_1'] = scores['bleu_1']['score']
    if 'bleu_4' in scores:
        metrics['BLEU_4'] = scores['bleu_4']['score']
    
    # Extract ROUGE score
    if 'rouge' in scores:
        metrics['ROUGE_L'] = scores['rouge']['rougeL']
    
    # Extract CIDEr score
    if 'cider' in scores:
        metrics['CIDEr'] = scores['cider']['score']
    
    # Compute BERTScore    
    P, R, F1 = score_bert([pred_text],
                         [gt_text_list],
                         lang='en',
                         rescale_with_baseline=True,
                         idf=True,
                         device=device,
                         verbose=False)
    
    metrics['BERTScore_F1'] = F1.mean().item()
    
    return metrics


def compute_and_save_unified_comparison_with_metrics(e2e_data, mgpt3_data, gt_data, output_path):
    """Save unified comparison with per-sample metrics to a .txt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get common sample indices (sorted)
    common_samples = sorted(set(e2e_data.keys()) & set(mgpt3_data.keys()) & set(gt_data.keys()))
    
    print(f"Computing metrics for {len(common_samples)} samples...")
    
    # Initialize NLG evaluator
    nlg_evaluator = initialize_nlg_evaluator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For aggregate metrics
    e2e_metrics_agg = {
        'BLEU_1': [], 'BLEU_4': [], 'ROUGE_L': [], 'CIDEr': [], 'BERTScore_F1': []
    }
    mgpt3_metrics_agg = {
        'BLEU_1': [], 'BLEU_4': [], 'ROUGE_L': [], 'CIDEr': [], 'BERTScore_F1': []
    }
    
    with open(output_path, 'w') as f:
        for sample_idx in tqdm(common_samples, desc="Processing samples"):
            f.write(f"{'='*80}\n")
            f.write(f"Sample: {sample_idx}\n")
            f.write(f"{'='*80}\n")
            
            # Get data
            pred_e2e = e2e_data[sample_idx]
            pred_mgpt3 = mgpt3_data[sample_idx]
            gt = gt_data[sample_idx]
            
            # Write predictions and ground truth
            f.write(f"\n--> Pred (ours-e2e):\n")
            f.write(f"    {pred_e2e}\n\n")
            
            f.write(f"--> Pred (ours-mgpt3):\n")
            f.write(f"    {pred_mgpt3}\n\n")

            # Compute metrics for e2e
            e2e_metrics = compute_single_sample_metrics(pred_e2e, gt, nlg_evaluator, device)
            f.write(f"--- Metrics (ours-e2e) ---\n")
            for metric_name, value in e2e_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
                e2e_metrics_agg[metric_name].append(value)
            f.write(f"\n")
            
            # Compute metrics for mgpt3
            mgpt3_metrics = compute_single_sample_metrics(pred_mgpt3, gt, nlg_evaluator, device)
            f.write(f"--- Metrics (ours-mgpt3) ---\n")
            for metric_name, value in mgpt3_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
                mgpt3_metrics_agg[metric_name].append(value)
            f.write(f"\n")
            
            # Handle both string and list of strings for ground truth
            if isinstance(gt, list):
                for j, gt_item in enumerate(gt):
                    f.write(f"--> GT {j+1:02d}:\n")
                    f.write(f"    {gt_item}\n")
            else:
                f.write(f"--> GT 01:\n")
                f.write(f"    {gt}\n")
            
            f.write(f"\n")
        
        # Write aggregate statistics
        # f.write(f"\n{'='*80}\n")
        # f.write(f"AGGREGATE METRICS (Mean over {len(common_samples)} samples)\n")
        # f.write(f"{'='*80}\n\n")
        
        # f.write(f"--- E2E Model ---\n")
        # for metric_name, values in e2e_metrics_agg.items():
        #     if len(values) > 0:
        #         mean_val = np.mean(values)
        #         std_val = np.std(values)
        #         f.write(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        # f.write(f"\n--- MGPT3 Model ---\n")
        # for metric_name, values in mgpt3_metrics_agg.items():
        #     if len(values) > 0:
        #         mean_val = np.mean(values)
        #         std_val = np.std(values)
        #         f.write(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        # f.write(f"\n{'='*80}\n")
    
    print(f"\nUnified comparison with metrics saved to: {output_path}")
    print(f"Total matched samples: {len(common_samples)}")
    
    return e2e_metrics_agg, mgpt3_metrics_agg


if __name__ == "__main__":
    
    # # Process load_our_text_prediction for humanml
    # print("=" * 60)
    # print("Processing load_our_text_prediction for humanml dataset...")
    # print("=" * 60)
    # pred_texts_1, gt_texts_1 = load_our_text_prediction_humanml()
    # output_path_1 = "./debug_output/humanml_our_text_prediction_comparison.txt"
    # save_text_comparison(pred_texts_1, gt_texts_1, output_path_1)
    
    # # Process load_our_motion_motiongpt3 for humanml
    # print("\n" + "=" * 60)
    # print("Processing load_our_motion_motiongpt3 for humanml dataset...")
    # print("=" * 60)
    # pred_texts_2, gt_texts_2 = load_our_motion_motiongpt3_humanml()
    # output_path_2 = "./debug_output/humanml_our_motion_motiongpt3_comparison.txt"
    # save_text_comparison(pred_texts_2, gt_texts_2, output_path_2)

    # Process unified comparison
    print("\n" + "=" * 60)
    print("Processing unified comparison for humanml dataset...")
    print("=" * 60)
    e2e_data, mgpt3_data, gt_data = load_and_unify_predictions_humanml()
    # output_path_3 = "./debug_output/humanml_unified_comparison.txt"
    # save_unified_comparison(e2e_data, mgpt3_data, gt_data, output_path_3)
    output_path_4 = "./debug_output/humanml_unified_comparison_with_metrics.txt"
    e2e_metrics_agg, mgpt3_metrics_agg = compute_and_save_unified_comparison_with_metrics(
        e2e_data, mgpt3_data, gt_data, output_path_4
    )
    
    print("\n" + "=" * 60)
    print("Done! Both comparison files generated.")
    print("=" * 60)