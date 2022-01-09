python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NUM_DIGITS_CLASSIFIER_ON 1 \
        OUTPUT_DIR "./output/ablations/num_digit_cls_position/1"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NUM_DIGITS_CLASSIFIER_ON 2 \
        OUTPUT_DIR "./output/ablations/num_digit_cls_position/2"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NUM_DIGITS_CLASSIFIER_ON 3 \
        OUTPUT_DIR "./output/ablations/num_digit_cls_position/3"