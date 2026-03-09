from typing import List
import torch
from torchmetrics import Metric
from bert_score import score as score_bert
from tqdm import tqdm 
import numpy as np
import pickle
import os
import argparse
from omegaconf import OmegaConf
import glob


class M2TMetrics(Metric):

    def __init__(self,
                 cfg,
                 dataname='humanml3d',
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "NLG metrics"

        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []

        # NLG metrics
        self.add_state("ROUGE_L",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("ROUGE_L")

        self.add_state("CIDEr",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("CIDEr")

        # Cached texts
        self.pred_texts = []
        self.gt_texts = []

        # NLG Evaluator
        if self.cfg.model.params.task == 'm2t':
            from nlgmetricverse import NLGMetricverse, load_metric
            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            self.nlg_evaluator = NLGMetricverse(metrics)

    @torch.no_grad()
    def compute(self, sanity_flag):
        count_seq = self.count_seq.item()

        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        print("Computing text metrics...")

        # NLP metrics
        scores = self.nlg_evaluator(predictions=self.pred_texts,
                                    references=self.gt_texts)
        for key in scores.keys():
            if 'bleu' in key:
                metrics[key] = torch.tensor(scores[key]['score'], device=self.device)
            
        metrics["ROUGE_L"] = torch.tensor(scores["rouge"]["rougeL"],
                                          device=self.device)
        metrics["CIDEr"] = torch.tensor(scores["cider"]['score'], device=self.device)

        # Bert metrics
        batch_size = 64
        P_scores, R_scores, F1_scores = [], [], []
        print(f"Computing BERTScore in batches of {batch_size}...")
        for i in range(0, len(self.pred_texts), batch_size):
            batch_pred = self.pred_texts[i:i+batch_size]
            batch_gt = self.gt_texts[i:i+batch_size]
            
            P, R, F1 = score_bert(batch_pred,
                                batch_gt,
                                lang='en',
                                rescale_with_baseline=True,
                                idf=True,
                                device=self.device,
                                verbose=False)
            
            P_scores.append(P)
            R_scores.append(R)
            F1_scores.append(F1)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics["Bert_F1"] = torch.cat(F1_scores).mean()

        # P, R, F1 = score_bert(self.pred_texts,
        #                       self.gt_texts,
        #                       lang='en',
        #                       rescale_with_baseline=True,
        #                       idf=True,
        #                       device=self.device,
        #                       verbose=False)

        # metrics["Bert_F1"] = F1.mean()

        # Reset
        self.reset()
        self.gt_texts = []
        self.pred_texts = []

        return {**metrics}

    @torch.no_grad()
    def update(self,
               pred_texts: List[str],
               gt_texts: List[str]):

        self.count_seq += len(pred_texts)

        self.pred_texts.extend(pred_texts)
        self.gt_texts.extend(gt_texts)


def eval_and_save(metric, pred_texts, gt_texts, output_path):
    """Evaluate metrics and save results to a text file."""

    metric.update(pred_texts=pred_texts, gt_texts=gt_texts)

    results = metric.compute(sanity_flag=False)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Text Generation Metrics\n")
        f.write("=" * 50 + "\n")
        for metric_name, value in results.items():
            if isinstance(value, torch.Tensor):
                line = f"{metric_name}: {value.item():.4f}\n"
            else:
                line = f"{metric_name}: {value}\n"
            f.write(line)
            print(line.strip())
        f.write("=" * 50 + "\n")
    
    print(f"\nResults saved to: {output_path}")
    return results


def load_our_text_prediction(dataset):
    """Load our direct text predictions from .npy result files."""
    
    if dataset == 'parahome':
        pred_dir = "/scratch/bczy/tcheng1/exp_test_parahome/viz_test_generate_number_merged"
        files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
        
        pred_texts = []
        gt_texts = []
        for file in tqdm(files, desc="Loading predictions and ground truth"):
            data = np.load(file, allow_pickle=True).item()
            pred_text = data['pred']['description']
            gt_text = data['gt']['description']
            pred_texts.append(pred_text)
            gt_texts.append(gt_text)
    
    elif dataset == 'humoto':
        # pred_dir = "/work/hdd/bczy/tcheng1/exp_test_humoto/viz_test_generate_number_merged"
        pred_dir = "/scratch/bfyo/tcheng1/exp_test_finetuned_humoto_6imu_1000frame/viz_test_generate_number_original"
        gt_folder = "/scratch/bfyo/tcheng1/dataset_process/humoto_data/all"
        files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
        
        pred_texts = []
        gt_texts = []
        for file in tqdm(files, desc="Loading predictions and ground truth"):
            data = np.load(file, allow_pickle=True).item()
            sample_idx = data['sample_idx'][0]
            pred_text = data['pred']['description']
            
            sample = pickle.load(open(f"{gt_folder}/{sample_idx:07d}.pkl", "rb"))
            gt_text = sample['text']  # list of text
            
            pred_texts.append(pred_text)
            gt_texts.append(gt_text)
    
    elif dataset == 'humanml':
        # pred_dir = "/work/hdd/bczy/tcheng1/exp_test_humanml_6imu_60frame/viz_test_generate_number"
        # pred_dir = "/scratch/bfyo/tcheng1/exp_test_finetuned_humanml_6imu_1000frame/viz_test_generate_number_original"
        pred_dir = "/scratch/bfyo/tcheng1/exp_test_finetuned_humanml_6imu_120frame/viz_test_generate_number_shifted_0"
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
    
    elif dataset == 'lingo':
        # pred_dir = "/work/hdd/bczy/tcheng1/exp_test_lingo_6imu_2000frame/viz_test_generate_number_merged"
        pred_dir = "/work/hdd/bfyo/tcheng1/exp_test_finetuned_lingo_6imu_120frame/viz_test_generate_number_original"
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


def load_our_motion_motiongpt3(dataset):
    """Load our motion + MotionGPT3 text predictions."""
    if dataset == 'humanml':
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
    
    elif dataset == 'lingo':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_text_pred_from_scratch"
        gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_lingo_6imu_2000frame/viz_test_generate_number_merged"
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
    
    elif dataset == 'humoto':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humoto_text_pred_from_scratch"
        gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_humoto/viz_test_generate_number_merged"
        test_dataset_dir = '/scratch/bfyo/tcheng1/dataset_process/humoto_data/all'
        
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
            with open(os.path.join(test_dataset_dir, f"{test_sample_id:07d}.pkl"), 'rb') as f:
                data = pickle.load(f)
            gt_text = data['text']
            gt_texts.append(gt_text)
    
    elif dataset == 'parahome':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_parahome_text_pred_from_scratch"
        gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_parahome/viz_test_generate_number_merged"
        
        number_of_samples = len(os.listdir(pred_texts_dir))
        
        pred_texts = []
        for i in tqdm(range(number_of_samples), desc="Loading predictions"):
            with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
                pred_text = f.read().strip()
            pred_texts.append(pred_text)
        
        gt_texts = []
        for i in tqdm(range(number_of_samples), desc="Loading ground truth"):
            gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
            gt_text = gt_info['gt']['description']
            gt_texts.append(gt_text)
    
    return pred_texts, gt_texts


def load_mobileposer_motiongpt3(dataset):
    """Load MobilePoser motion + MotionGPT3 text predictions."""
    if dataset == 'humanml':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_text_pred_from_scratch"
        gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/humanml_gt_text.pkl"
    elif dataset == 'lingo':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_text_pred_from_scratch"
        gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/LINGO_gt_text.pkl"
    elif dataset == 'humoto':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humoto_text_pred_from_scratch"
        gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/HUMOTO_gt_text.pkl"
    elif dataset == 'parahome':
        pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_parahome_text_pred_from_scratch"
        gt_texts_file = "/scratch/benk/tcheng1/gt_text/ParaHome_gt_text.pkl"
    
    with open(gt_texts_file, 'rb') as f:
        gt_texts_dict = pickle.load(f)

    number_of_samples = len(gt_texts_dict)
    
    if dataset == 'parahome':
        # TODO: may need update
        gt_texts = [gt_texts_dict[i] for i in range(number_of_samples)]
    elif dataset in ['humanml', 'lingo', 'humoto']:
        gt_texts = []
        gt_data_path = [gt_texts_dict[i] for i in range(number_of_samples)]
        for data_path in tqdm(gt_data_path, desc="Loading ground truth"):

            # HUMOTO dataset: 
            # from /scratch/benk/tcheng1/code/imu-human-mllm/dataset_process/humoto_data/all 
            # into /scratch/bfyo/tcheng1/dataset_process/humoto_data/all
            if dataset == 'humoto':
                data_path = data_path.replace('/scratch/benk/tcheng1/code/imu-human-mllm/dataset_process/humoto_data/all',
                                              '/scratch/bfyo/tcheng1/dataset_process/humoto_data/all')

            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                if 'texts' in data:
                    gt_text = data['texts']
                elif 'text' in data:
                    gt_text = data['text']
                else:
                    raise ValueError("Ground truth text key not found.")
                gt_texts.append(gt_text)
    
    pred_texts = []
    for i in tqdm(range(number_of_samples), desc="Loading predictions"):
        with open(os.path.join(pred_texts_dir, f"sample_seq_{i:04d}.txt"), 'r') as f:
            pred_text = f.read().strip()
        pred_texts.append(pred_text)
    
    return pred_texts, gt_texts




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate text generation metrics')
    parser.add_argument('--setting', type=str, required=False,
                        choices=['ours_text', 'ours_mgpt3', 'mobileposer_mgpt3'])
    parser.add_argument('--dataset', type=str, required=False,
                        choices=['parahome', 'humoto', 'humanml', 'lingo'])
    parser.add_argument('--out_dir', type=str, required=False, default='/home/haoyuyh3/Documents/maxhsu/imu-humans/metric_results/text',
                        help='Directory to save metric results')
    args = parser.parse_args()

    print(f"Scenario: {args.setting}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.out_dir}")
    
    # Create a mock config 
    cfg = OmegaConf.create({
        'model': {
            'params': {
                'task': 'm2t'
            }
        }
    })
    
    # Initialize metrics
    metric = M2TMetrics(cfg=cfg, dataname='humanml3d')
    metric = metric.cuda() if torch.cuda.is_available() else metric

    # # Load data based on scenario
    # print("\nLoading data...")
    # if args.setting == 'ours_text':
    #     pred_texts, gt_texts = load_our_text_prediction(args.dataset)
    # elif args.setting == 'ours_mgpt3':
    #     pred_texts, gt_texts = load_our_motion_motiongpt3(args.dataset)
    # elif args.setting == 'mobileposer_mgpt3':
    #     pred_texts, gt_texts = load_mobileposer_motiongpt3(args.dataset)

    # assert len(pred_texts) == len(gt_texts), "Number of predictions and ground truths must match"

    # out_fname = f"{args.setting}_{args.dataset}_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)


    ######################################################################
    # TODO: manually loading and evaluating for custom model settings
    ######################################################################

    pred_texts_dir = "/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_imuposer_smplx/lingo_global/text_pred_mgpt3"
    gt_texts_dir = "/home/haoyuyh3/Downloads/lingo_smpl_files/test"

    fname_list = sorted([f for f in os.listdir(gt_texts_dir) if f.endswith('.pkl')])

    gt_texts, pred_texts = [], []
    for i, fname in tqdm(enumerate(fname_list), dynamic_ncols=True):

        name = fname.split('.')[0]  # e.g., "xxx.pkl" -> "xxx"

        with open(os.path.join(pred_texts_dir, f"{name}.txt"), 'r') as f:
            pred_text = f.read().strip()
        
        gt_file_path = os.path.join(gt_texts_dir, fname)
        with open(gt_file_path, 'rb') as f:
            data = pickle.load(f)
            if 'texts' in data:
                gt_text = data['texts']
            elif 'text' in data:
                gt_text = data['text']
            else:
                raise ValueError("Ground truth text key not found.")
        
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)

    print("\nEvaluating metrics...")
    out_path = os.path.join('/home/haoyuyh3/Documents/maxhsu/imu-humans/metric_results/text', 'imuposer_smplx_mgpt3_lingo_metrics.txt')
    results = eval_and_save(metric, pred_texts, gt_texts, out_path)



    pred_texts_dir = "/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_data/_pred_imuposer_smplx/lingo_global/text_pred_mgpt"
    gt_texts_dir = "/home/haoyuyh3/Downloads/lingo_smpl_files/test"

    fname_list = sorted([f for f in os.listdir(gt_texts_dir) if f.endswith('.pkl')])

    gt_texts, pred_texts = [], []
    for i, fname in tqdm(enumerate(fname_list), dynamic_ncols=True):

        name = fname.split('.')[0]  # e.g., "xxx.pkl" -> "xxx"

        with open(os.path.join(pred_texts_dir, f"{name}.txt"), 'r') as f:
            pred_text = f.read().strip()
        
        gt_file_path = os.path.join(gt_texts_dir, fname)
        with open(gt_file_path, 'rb') as f:
            data = pickle.load(f)
            if 'texts' in data:
                gt_text = data['texts']
            elif 'text' in data:
                gt_text = data['text']
            else:
                raise ValueError("Ground truth text key not found.")
        
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)

    print("\nEvaluating metrics...")
    out_path = os.path.join('/home/haoyuyh3/Documents/maxhsu/imu-humans/metric_results/text', 'imuposer_smplx_mgpt_lingo_metrics.txt')
    results = eval_and_save(metric, pred_texts, gt_texts, out_path)



    # # IMUPoser + MotionGPT3 text pred on LINGO test set
    # pred_texts_dir = "/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_lingo_text_pred_mgpt3"
    # gt_texts_dir = "/home/haoyuyh3/Downloads/lingo_smpl_files/test"

    # with open('/home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/lingo_test_fname.txt', 'r') as f:
    #     gt_fname_list = f.read().splitlines()

    # gt_texts, pred_texts = [], []
    # for i in tqdm(range(len(os.listdir(pred_texts_dir))), dynamic_ncols=True):
    #     with open(os.path.join(pred_texts_dir, f"sample_{i:05d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
        
    #     gt_file_path = os.path.join(gt_texts_dir, gt_fname_list[i])
    #     with open(gt_file_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts']
    #         elif 'text' in data:
    #             gt_text = data['text']
    #         else:
    #             raise ValueError("Ground truth text key not found.")
        
    #     pred_texts.append(pred_text)
    #     gt_texts.append(gt_text)

    # out_fname = f"imuposer_mgpt3_lingo_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)


    # # IMUPoser + MotionGPT text pred on LINGO test set
    # pred_texts_dir = "/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_lingo_text_pred_mgpt"
    # gt_texts_dir = "/home/haoyuyh3/Downloads/lingo_smpl_files/test"

    # with open('/home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/lingo_test_fname.txt', 'r') as f:
    #     gt_fname_list = f.read().splitlines()

    # gt_texts, pred_texts = [], []
    # for i in tqdm(range(len(os.listdir(pred_texts_dir))), dynamic_ncols=True):
    #     with open(os.path.join(pred_texts_dir, f"sample_{i:05d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
        
    #     gt_file_path = os.path.join(gt_texts_dir, gt_fname_list[i])
    #     with open(gt_file_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts']
    #         elif 'text' in data:
    #             gt_text = data['text']
    #         else:
    #             raise ValueError("Ground truth text key not found.")
        
    #     pred_texts.append(pred_text)
    #     gt_texts.append(gt_text)

    # out_fname = f"imuposer_mgpt_lingo_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)


    # # IMUPoser + MotionGPT3 text pred on HumanML test set
    # pred_texts_dir = "/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_humanml_text_pred_mgpt3"
    # gt_texts_dir = "/home/haoyuyh3/Downloads/humanml_smpl_files/test"

    # with open('/home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/humanml_test_fname.txt', 'r') as f:
    #     gt_fname_list = f.read().splitlines()

    # gt_texts, pred_texts = [], []
    # for i in tqdm(range(len(os.listdir(pred_texts_dir))), dynamic_ncols=True):
    #     with open(os.path.join(pred_texts_dir, f"sample_{i:05d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
        
    #     gt_file_path = os.path.join(gt_texts_dir, gt_fname_list[i])
    #     with open(gt_file_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts']
    #         elif 'text' in data:
    #             gt_text = data['text']
    #         else:
    #             raise ValueError("Ground truth text key not found.")
        
    #     pred_texts.append(pred_text)
    #     gt_texts.append(gt_text)

    # out_fname = f"imuposer_mgpt3_humanml_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)


    # # IMUPoser + MotionGPT text pred on HumanML test set
    # pred_texts_dir = "/home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_humanml_text_pred_mgpt"
    # gt_texts_dir = "/home/haoyuyh3/Downloads/humanml_smpl_files/test"

    # with open('/home/haoyuyh3/Documents/maxhsu/imu-humans/IMUPoser/humanml_test_fname.txt', 'r') as f:
    #     gt_fname_list = f.read().splitlines()

    # gt_texts, pred_texts = [], []
    # for i in tqdm(range(len(os.listdir(pred_texts_dir))), dynamic_ncols=True):
    #     with open(os.path.join(pred_texts_dir, f"sample_{i:05d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
        
    #     gt_file_path = os.path.join(gt_texts_dir, gt_fname_list[i])
    #     with open(gt_file_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts']
    #         elif 'text' in data:
    #             gt_text = data['text']
    #         else:
    #             raise ValueError("Ground truth text key not found.")
        
    #     pred_texts.append(pred_text)
    #     gt_texts.append(gt_text)

    # out_fname = f"imuposer_mgpt_humanml_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)


    # MobilePoser + MotionGPT text pred on LINGO test set
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_mgpt_lingo_text_pred_from_scratch"
    # gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/LINGO_gt_text.pkl"
    
    # with open(gt_texts_file, 'rb') as f:
    #     gt_texts_dict = pickle.load(f)

    # number_of_samples = len(gt_texts_dict)

    # gt_texts = []
    # gt_data_path = [gt_texts_dict[i] for i in range(number_of_samples)]
    # for data_path in tqdm(gt_data_path, desc="Loading ground truth"):

    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts']
    #         elif 'text' in data:
    #             gt_text = data['text']
    #         else:
    #             raise ValueError("Ground truth text key not found.")
    #         gt_texts.append(gt_text)
    
    # pred_texts = []
    # for i in tqdm(range(number_of_samples), desc="Loading predictions"):
    #     with open(os.path.join(pred_texts_dir, f"sample_seq_{i:04d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #     pred_texts.append(pred_text)

    # out_fname = f"mobileposer_mgpt_lingo_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)



    # # MobilePoser + MotionGPT text pred on HumanML test set
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_mgpt_humanml_text_pred_from_scratch"
    # gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/humanml_gt_text.pkl"
    
    # with open(gt_texts_file, 'rb') as f:
    #     gt_texts_dict = pickle.load(f)

    # number_of_samples = len(gt_texts_dict)

    # gt_texts = []
    # gt_data_path = [gt_texts_dict[i] for i in range(number_of_samples)]
    # for data_path in tqdm(gt_data_path, desc="Loading ground truth"):

    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts']
    #         elif 'text' in data:
    #             gt_text = data['text']
    #         else:
    #             raise ValueError("Ground truth text key not found.")
    #         gt_texts.append(gt_text)
    
    # pred_texts = []
    # for i in tqdm(range(number_of_samples), desc="Loading predictions"):
    #     with open(os.path.join(pred_texts_dir, f"sample_seq_{i:04d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #     pred_texts.append(pred_text)

    # out_fname = f"mobileposer_mgpt_humanml_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)



    # Ours (IMU-Text) without motion modality on HumanML test set (NOTE: there is only 19k samples, may need re-evaluation)
    # pred_dir = "/scratch/bfyo/tcheng1/exp_text_only_test_humanml_6imu_2000frame/viz_test_generate_number_original"
    # gt_folder = "/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence/motion_data/test"
    # files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))
    
    # pred_texts = []
    # gt_texts = []
    # for file in tqdm(files, desc="Loading predictions and ground truth"):
    #     data = np.load(file, allow_pickle=True).item()
    #     sample_idx = data['sample_idx'][0]
    #     pred_text = data['pred']['description']
        
    #     sample = pickle.load(open(f"{gt_folder}/{sample_idx}.pkl", "rb"))
    #     gt_text = sample['texts']  # list of text
        
    #     pred_texts.append(pred_text)
    #     gt_texts.append(gt_text)

    # out_fname = f"ours_only_text_humanml_metrics.txt"
    # out_path = os.path.join(args.out_dir, out_fname)

    # print("\nEvaluating metrics...")
    # results = eval_and_save(metric, pred_texts, gt_texts, out_path)


    




    ##### MobilePoser motions + MotionGPT3 text predictions #####
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humoto_text_pred"
    # gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/HUMOTO_gt_text.pkl"
    
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_text_pred"
    # gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/LINGO_gt_text.pkl"

    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_text_pred"
    # gt_texts_file = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/gt_text/humanml_gt_text.pkl"

    # with open(gt_texts_file, 'rb') as f:
    #     gt_texts_dict = pickle.load(f)

    # number_of_samples = len(os.listdir(pred_texts_dir))
    # print("Number of samples:", number_of_samples)

    # pred_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     with open(os.path.join(pred_texts_dir, f"sample_seq_{i:04d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #         pred_texts.append(pred_text)

    # gt_texts = []
    # gt_data_path = [gt_texts_dict[i] for i in range(number_of_samples)]
    # for data_path in tqdm(gt_data_path):
    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)
    #         if 'texts' in data:
    #             gt_text = data['texts'][0]
    #         elif 'text' in data:
    #             gt_text = data['text'][0]
    #         else:
    #             raise ValueError("Ground truth text key not found.")
    #         gt_texts.append(gt_text)

    # assert len(pred_texts) == len(gt_texts), "Number of predictions and ground truths must match."



    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_parahome_text_pred"
    # gt_texts_file = "/scratch/benk/tcheng1/gt_text/ParaHome_gt_text.pkl"
    
    # with open(gt_texts_file, 'rb') as f:
    #     gt_texts_dict = pickle.load(f)

    # gt_texts = [gt_texts_dict[i] for i in range(len(gt_texts_dict))]

    # number_of_samples = len(gt_texts_dict)
    # print("Number of samples:", number_of_samples)

    # pred_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     with open(os.path.join(pred_texts_dir, f"sample_seq_{i:04d}.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #         pred_texts.append(pred_text)

    # assert len(pred_texts) == len(gt_texts), "Number of predictions and ground truths must match."



    ##### Our text predictions #####
    # print('ParaHome text metrics:')
    # with open('/scratch/benk/tcheng1/code/imu-human-mllm/third_party/Showo/calculate_metric/text/Ours_ParaHome_text_result.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # pred_texts = []
    # gt_texts = []
    # for k, v in data.items():
    #     pred_texts.append(v['pred_text']) # str
    #     gt_texts.append(v['gt_text'][0]) # list of strings, take the first one


    # print('HUMOTO text metrics:')
    # with open('/scratch/benk/tcheng1/code/imu-human-mllm/third_party/Showo/calculate_metric/text/Ours_HUMOTO_text_result.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # pred_texts = []
    # gt_texts = []
    # for k, v in data.items():
    #     pred_texts.append(v['pred_text']) # str
    #     gt_texts.append(v['gt_text'][0]) # list of strings, take the first one


    # print('LINGO text metrics:')
    # with open('/scratch/benk/tcheng1/code/imu-human-mllm/third_party/Showo/calculate_metric/text/Ours_LINGO_text_result.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # pred_texts = []
    # gt_texts = []
    # for k, v in data.items():
    #     pred_texts.append(v['pred_text']) # str
    #     gt_texts.append(v['gt_text'][0]) # list of strings, take the first one


    # print('HumanML text metrics:')   # only got 3703 samples
    # with open('/scratch/benk/tcheng1/code/imu-human-mllm/third_party/Showo/calculate_metric/text/Ours_HUMANML_text_result.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # pred_texts = []
    # gt_texts = []
    # for k, v in data.items():
    #     pred_texts.append(v['pred_text']) # str
    #     gt_texts.append(v['gt_text'][0]) # list of strings, take the first one



    # ##### Our motions + MotionGPT3 text predictions #####
    ### HUMANML ###
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_text_pred"
    # gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_humanml_6imu_60frame/viz_test_generate_number"
    # test_dataset_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence/motion_data/test'
    
    # number_of_samples = len(os.listdir(pred_texts_dir)[:10])

    # pred_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #     pred_texts.append(pred_text)

    # gt_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
    #     test_sample_id = gt_info['sample_idx'][0]
    #     with open(os.path.join(test_dataset_dir, f"{test_sample_id}.pkl"), 'rb') as f:
    #         data = pickle.load(f)
    #     gt_text = data['texts']
    #     gt_texts.append(gt_text)


    ### LINGO ###
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_text_pred"
    # gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_lingo_6imu_2000frame/viz_test_generate_number_merged"
    # test_dataset_dir = '/work/hdd/benk/hhsu2/imu-humans/final_data_per_sequence/motion_data/test'

    # number_of_samples = len(os.listdir(pred_texts_dir))
    
    # pred_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #     pred_texts.append(pred_text)

    # gt_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
    #     test_sample_id = gt_info['sample_idx'][0]
    #     with open(os.path.join(test_dataset_dir, f"{test_sample_id}.pkl"), 'rb') as f:
    #         data = pickle.load(f)
    #     gt_text = data['texts'][0]
    #     gt_texts.append(gt_text)


    ### HUMOTO ###
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humoto_text_pred"
    # gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_humoto/viz_test_generate_number_merged"
    # test_dataset_dir = '/scratch/benk/tcheng1/code/imu-human-mllm/dataset_process/humoto_data/all'

    # number_of_samples = len(os.listdir(pred_texts_dir))
    
    # pred_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #     pred_texts.append(pred_text)

    # gt_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
    #     test_sample_id = gt_info['sample_idx'][0]
    #     with open(os.path.join(test_dataset_dir, f"{test_sample_id:07d}.pkl"), 'rb') as f:
    #         data = pickle.load(f)
    #     gt_text = data['text'][0]
    #     gt_texts.append(gt_text)


    ### PARAHOME ###
    # pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_parahome_text_pred"
    # gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_parahome/viz_test_generate_number_merged"

    # number_of_samples = len(os.listdir(pred_texts_dir))
    
    # pred_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
    #         pred_text = f.read().strip()
    #     pred_texts.append(pred_text)

    # gt_texts = []
    # for i in tqdm(range(number_of_samples)):
    #     gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
    #     gt_text = gt_info['gt']['description'][0]
    #     gt_texts.append(gt_text)


