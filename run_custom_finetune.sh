# Finetune MotionGPT3 from the pretrained checkpoint
python -m train \
    --cfg configs/MoT_vae_custom_finetune_humanml.yaml \
    --nodebug \
    --pretrained /projects/benk/hhsu2/imu-humans/related_works/MotionGPT3/checkpoints/motiongpt3.ckpt


python -m train \
    --cfg configs/MoT_vae_custom_finetune_lingo.yaml \
    --nodebug \
    --pretrained /projects/benk/hhsu2/imu-humans/related_works/MotionGPT3/checkpoints/motiongpt3.ckpt