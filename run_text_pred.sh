# Note: demo_pretrained_ours.py and demo_pretrained_mobileposer.py are deprecated for now.


##### Train from scratch on MotionGPT3 models #####

# python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_humanml.yaml \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_text_pred_from_scratch


# python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_humanml.yaml \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_text_pred_from_scratch


# python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_lingo.yaml \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_text_pred_from_scratch


# python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_lingo.yaml \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_text_pred_from_scratch


# python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_humanml.yaml \
#     --input_263_dim_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_humanml_263d \
#     --output_text_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_humanml_text_pred_mgpt3


python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_lingo.yaml \
    --input_263_dim_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_lingo_263d \
    --output_text_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_lingo_text_pred_mgpt3


##### Train from scratch on MotionGPT models #####

# python demo_custom_ckpts_ours.py --cfg ./configs/mgpt_vae_custom_from_scratch_humanml.yaml \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_mgpt_humanml_text_pred_from_scratch


# python demo_custom_ckpts_ours.py --cfg ./configs/mgpt_vae_custom_from_scratch_humanml.yaml \
#     --input_263_dim_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_humanml_263d \
#     --output_text_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_humanml_text_pred_mgpt


python demo_custom_ckpts_ours.py --cfg ./configs/mgpt_vae_custom_from_scratch_lingo.yaml \
    --input_263_dim_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_lingo_263d \
    --output_text_dir /home/haoyuyh3/Documents/maxhsu/imu-humans/_tmp_pred_imuposer_lingo_text_pred_mgpt