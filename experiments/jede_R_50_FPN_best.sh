python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION 56 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS 3,1,1 \
        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
        DATASETS.TEST_VIDEO_IDS [0] \
        OUTPUT_DIR "./output/jede_R_50_FPN_best/test_0"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION 56 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS 3,1,1 \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        OUTPUT_DIR "./output/jede_R_50_FPN_best/test_1"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION 56 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS 3,1,1 \
        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
        DATASETS.TEST_VIDEO_IDS [2] \
        OUTPUT_DIR "./output/jede_R_50_FPN_best/test_2"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION 56 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS 3,1,1 \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
        DATASETS.TEST_VIDEO_IDS [3] \
        OUTPUT_DIR "./output/jede_R_50_FPN_best/test_3"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION 56 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS 3,1,1 \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        DATASETS.TEST_VIDEO_IDS [4] \
        OUTPUT_DIR "./output/jede_R_50_FPN_best/test_4"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL 1 \
        MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION 56 \
        MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS 3,1,1 \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        INPUT.AUG.COPY_PASTE_MIX 0 \
        OUTPUT_DIR "./output/jede_R_50_FPN_best/test_5"
