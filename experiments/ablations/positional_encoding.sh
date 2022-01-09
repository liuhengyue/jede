python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_NECK_BASE.PE True \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/pe/neck_base"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.PE True \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/pe/keypoints_branch"