python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.ROI_NECK_BASE.FUSION_TYPE "cat" \
        OUTPUT_DIR "./output/ablations/fusion/cat"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.ROI_NECK_BASE.FUSION_TYPE "sum" \
        OUTPUT_DIR "./output/ablations/fusion/sum"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.ROI_NECK_BASE.FUSION_TYPE "multiply" \
        OUTPUT_DIR "./output/ablations/fusion/multiply"