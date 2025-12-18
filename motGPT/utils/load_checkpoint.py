import torch
import sys


def load_pretrained(cfg, model, logger=None, phase="train"):
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Filter out unexpected keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    unexpected_keys = checkpoint_keys - model_keys
    missing_keys = model_keys - checkpoint_keys
    
    if unexpected_keys:
        if logger is not None:
            logger.warning(f"Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")
        # Remove unexpected keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    
    if missing_keys:
        if logger is not None:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")

    model.load_state_dict(state_dict, strict=False)
    model.epoch = ckpt.get('epoch', -1)

    return model


def load_pretrained_vae(cfg, model, logger=None):
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE, weights_only=False,
                            map_location="cpu")['state_dict']
    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        # if 'skel_embedding' in k: continue
        # if 'final_layer' in k:continue
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v

    if hasattr(model, 'vae'):
        model.vae.load_state_dict(vae_dict, strict=True)
    else:
        model.motion_vae.load_state_dict(vae_dict, strict=True)
    
    return model
