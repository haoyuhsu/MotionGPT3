# Finetune MotionGPT3 from the pretrained checkpoint
python -m train \
    --cfg configs/MoT_vae_custom_finetune_humanml.yaml \
    --nodebug


python -m train \
    --cfg configs/MoT_vae_custom_finetune_lingo.yaml \
    --nodebug


python -m train \
    --cfg configs/MoT_vae_custom_finetune_lingo_debug.yaml \
    --nodebug