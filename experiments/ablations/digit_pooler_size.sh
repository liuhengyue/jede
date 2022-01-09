python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_BOX_HEAD.POOLER_RESOLUTION 7 \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/digit_pooler_size/7x7"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_BOX_HEAD.POOLER_RESOLUTION 14 \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/digit_pooler_size/14x14"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_BOX_HEAD.POOLER_RESOLUTION 28 \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/digit_pooler_size/28x28"