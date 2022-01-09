python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 4 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 4 \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/gt_spatial_size/28x28"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/gt_spatial_size/14x14"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 2 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 4 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 2 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 4 \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/gt_spatial_size/56x56"