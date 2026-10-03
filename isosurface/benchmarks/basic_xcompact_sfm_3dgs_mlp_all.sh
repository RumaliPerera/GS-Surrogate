RESULT_DIR="results/xcompact"
RENDER_TRAJ_PATH="ellipse"


FOLDER="SfM_3DGS_Deform_mlp_all"
DATASET="/path/to/XCompact_Dataset_Isosurfaces"


mkdir -p $RESULT_DIR/$FOLDER


CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH nohup python -u simple_trainer.py default \
    --data_dir $DATASET \
    --use_deformation --use_surface \
    --holdout_conditions "9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121" \
    --reference_condition 64 \
    --holdout_isovalues "" \
    \
    --surface_deform_feature_dim 128 \
    --surface_deform_hidden_dim 512 \
    \
    --deform_start_step 0 \
    --max_steps 150000 \
    --test_every 2 \
    --learn_deform_sh \
    --learn_deform_alpha \
    \
    --white_bkgd \
    --data_factor 1 \
    \
    --hard_mining_exponent 1.5 \
    \
    --surface_deform_scale 1.0 \
    --surface_deform_lr 0.0001 \
    --no-freeze_canonical_in_stage2 --stage2_canonical_lr_scale 0.001 \
    \
    --render_traj_path $RENDER_TRAJ_PATH \
    --result_dir $RESULT_DIR/$FOLDER/ \
    --disable_viewer > $RESULT_DIR/$FOLDER/log.log
