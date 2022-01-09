
python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_DIGIT_NECK_OUTPUT.MIN_OVERLAP 0.1 \
        OUTPUT_DIR "./output/ablations/guassian_target_radius/min_overlap_0.1"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_DIGIT_NECK_OUTPUT.MIN_OVERLAP 0.5 \
        OUTPUT_DIR "./output/ablations/guassian_target_radius/min_overlap_0.5"