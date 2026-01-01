import json
import os
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
import motGPT.render.matplot.plot_3d_global as plot_3d


def main():
    # parse options
    cfg, params = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # create logger
    logger = create_logger(cfg, phase="test")

    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # create model
    model = build_model(cfg, datamodule).eval()
    logger.info("model {} loaded".format(cfg.model.target))

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
        state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                                map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(
            "No checkpoints provided, using random initialized model")

    model.to(device)


    def predict_text_from_motion(motion_feats, lengths):

        with torch.no_grad():
            motion_tokens = model.lm.motion_feats_to_tokens(model.vae, motion_feats, lengths, modes='motion')

        tasks = [{
            'class': 'm2t',
            'input': ['Describe the motion represented by <Motion_Placeholder> using plain English.'],
            'output': ['']
        }] * len(lengths)
        texts = [''] * len(lengths)

        inputs, outputs, modes = model.lm.template_fulfill(tasks, lengths, texts)
        
        outputs_tokens, cleaned_text = model.lm.generate_direct(
            inputs,
            motion_tokens=motion_tokens,
            max_length=40,
            num_beams=1,
            do_sample=False,
            gen_mode='text',
        )
        return cleaned_text


    input_263_dim_dir = params.input_263_dim_dir
    output_text_dir = params.output_text_dir
    os.makedirs(output_text_dir, exist_ok=True)

    motion_feat_list = [os.path.join(input_263_dim_dir, f) for f in os.listdir(input_263_dim_dir) if f.endswith('.npy')]


    # id_xxxxx_step_0.npy
    # motion_feat_list = sorted(motion_feat_list, key=lambda x: int(os.path.basename(x).split('_')[1]))

    # sample_seq_xxxxx.npy
    # motion_feat_list = sorted(motion_feat_list, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))


    for motion_feat_path in tqdm(motion_feat_list, desc="Processing motions"):

        base_name = os.path.basename(motion_feat_path).replace('.npy', '.txt')
        output_text_path = os.path.join(output_text_dir, base_name)

        if os.path.exists(output_text_path):
            print(f"Output text for {motion_feat_path} already exists, skipping...")
            continue
        
        # Load and prepare
        motion_feats = torch.tensor(np.load(motion_feat_path), device=device)  # (motion_length, 263)
        motion_feats = model.datamodule.normalize(motion_feats)

        if motion_feats.shape[0] > 498:
            motion_feats = motion_feats[:498]   # truncate to max length 498 since this is the maximum length
        
        motion_feats = motion_feats.unsqueeze(0)  # (1, motion_length, 263)
        lengths = [motion_feats.shape[1]]  # list of lengths

        pred_text = predict_text_from_motion(motion_feats, lengths)
        print(f"Prediction for {motion_feat_path}: {pred_text}")

        # pred_text should be a list of length 1
        assert len(pred_text) == 1
        cleaned_text = pred_text[0].replace('"', '').lstrip().rstrip()
        print(f"Cleaned Prediction: {cleaned_text}")

        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)


if __name__ == "__main__":
    with torch.no_grad():
        main()
