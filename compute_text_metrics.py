from typing import List
import torch
from torchmetrics import Metric
from bert_score import score as score_bert
from tqdm import tqdm 
import numpy as np
import pickle
import os


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        P, R, F1 = score_bert(self.pred_texts,
                              self.gt_texts,
                              lang='en',
                              rescale_with_baseline=True,
                              idf=True,
                              device=self.device,
                              verbose=False)

        metrics["Bert_F1"] = F1.mean()

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


# Example usage
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
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
    
    # # Example ground truth and prediction texts
    # gt_texts = [
    #     "A person walks forward and then turns left.",
    #     "Someone is running quickly across the room.",
    #     "A person raises both arms above their head."
    # ]
    
    # pred_texts = [
    #     "A person is walking forward and turning to the left.",
    #     "A person runs fast across the space.",
    #     "Someone lifts their arms upward."
    # ]


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

    # number_of_samples = 3703  # len(os.listdir(pred_texts_dir))
    
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
    pred_texts_dir = "/projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_parahome_text_pred"
    gt_texts_dir = "/work/hdd/bczy/tcheng1/exp_test_parahome/viz_test_generate_number_merged"

    number_of_samples = len(os.listdir(pred_texts_dir))
    
    pred_texts = []
    for i in tqdm(range(number_of_samples)):
        with open(os.path.join(pred_texts_dir, f"id_{i}_step_0.txt"), 'r') as f:
            pred_text = f.read().strip()
        pred_texts.append(pred_text)

    gt_texts = []
    for i in tqdm(range(number_of_samples)):
        gt_info = np.load(os.path.join(gt_texts_dir, f"id_{i}_step_0.npy"), allow_pickle=True).item()
        gt_text = gt_info['gt']['description'][0]
        gt_texts.append(gt_text)



    # Update metrics with predictions and ground truth
    metric.update(pred_texts=pred_texts, gt_texts=gt_texts)
    
    # Compute final metrics
    results = metric.compute(sanity_flag=False)
    
    # Print results
    print("\n" + "="*50)
    print("Text Generation Metrics:")
    print("="*50)
    for metric_name, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{metric_name}: {value.item():.4f}")
        else:
            print(f"{metric_name}: {value}")
    print("="*50)






# RESULTS:
# Our motions + MotionGPT3 text predictions
# ##### HUMANML #####
# ==================================================
# Text Generation Metrics:
# ==================================================
# ROUGE_L: 0.2740
# CIDEr: 0.2401
# bleu_1: 0.2988
# bleu_4: 0.0517
# Bert_F1: 0.2364
# ==================================================
# ##### LINGO #####
# ==================================================
# Text Generation Metrics:
# ==================================================
# ROUGE_L: 0.0876
# CIDEr: 0.0958
# bleu_1: 0.0647
# bleu_4: 0.0000
# Bert_F1: 0.0284
# ==================================================
# ##### HUMOTO #####
# ==================================================
# Text Generation Metrics:
# ==================================================
# ROUGE_L: 0.1812
# CIDEr: 0.0845
# bleu_1: 0.1382
# bleu_4: 0.0154
# Bert_F1: 0.1069
# ==================================================
# ##### PARAHOME #####
# ==================================================
# Text Generation Metrics:
# ==================================================
# ROUGE_L: 0.0354
# CIDEr: 0.0000
# bleu_1: 0.0000
# bleu_4: 0.0000
# Bert_F1: -0.2954
# ==================================================