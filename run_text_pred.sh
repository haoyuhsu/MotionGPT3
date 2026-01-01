
python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_humanml.yaml \
    --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_263dim \
    --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_text_pred_from_scratch


python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_humanml.yaml \
    --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_263dim \
    --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_text_pred_from_scratch


python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_lingo.yaml \
    --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_263dim \
    --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_text_pred_from_scratch


python demo_custom_ckpts_ours.py --cfg ./configs/MoT_vae_custom_from_scratch_lingo.yaml \
    --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_263dim \
    --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_text_pred_from_scratch



# python demo_pretrained_mobileposer.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humoto_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humoto_text_pred


# python demo_pretrained_mobileposer.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_lingo_text_pred


# python demo_pretrained_mobileposer.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_parahome_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_parahome_text_pred


# python demo_pretrained_mobileposer.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_mobileposer/result_humanml_text_pred




# python demo_pretrained_ours.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humanml_text_pred


# python demo_pretrained_ours.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humoto_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_humoto_text_pred


# python demo_pretrained_ours.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_lingo_text_pred


# python demo_pretrained_ours.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt \
#     --input_263_dim_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_parahome_263dim \
#     --output_text_dir /projects/benk/hhsu2/imu-humans/related_works/Mocap-to-SMPLX/test_data_ours/result_parahome_text_pred