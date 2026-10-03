
RESULT_DIR="results/nyx"
RENDER_TRAJ_PATH="ellipse"


FOLDER="TF_adapter_20pct_100000iters_seed1"
DATASET="/path/to/Nyx_Dataset_TFs"
VOL_CKPT="/path/to/volume_model/ckpts/ckpt_109999_rank0.pt"


mkdir -p $RESULT_DIR/$FOLDER



CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH nohup python -u simple_trainer.py default \
    --data_dir $DATASET \
    \
    --use_deformation \
    --use_tf_adapter \
    --tf_file $DATASET/transfer_functions.txt \
    \
    --pretrained_volume_ckpt $VOL_CKPT \
    \
    --deform_feature_dim 128 \
    --deform_hidden_dim 512 \
    --deform_scale 1.0 \
    --learn_deform_sh \
    --learn_deform_alpha \
    \
    --tf_adapter_feature_dim 64 \
    --tf_adapter_hidden_dim 256 \
    --tf_adapter_alpha_scale 1.0 \
    --tf_adapter_sh_scale 1.0 \
    --tf_adapter_lr 0.0001 \
    --tf_adapter_num_layers 3 \
    --tf_adapter_reg 0.0 \
    \
    --tf_member_fraction 0.2 \
    --tf_member_seed 1 \
    \
    --holdout_conditions "100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129" \
    \
    --max_steps 100000 \
    --deform_start_step 0 \
    \
    --test_every 10 \
    --data_factor 1 \
    --hard_mining_exponent 1.5 \
    \
    --render_traj_path $RENDER_TRAJ_PATH \
    --result_dir $RESULT_DIR/$FOLDER/ \
    --disable_viewer > $RESULT_DIR/$FOLDER/log.log


