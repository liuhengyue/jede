#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
#        DATASETS.TEST_VIDEO_IDS [0]
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
#        DATASETS.TEST_VIDEO_IDS [1] \
#        OUTPUT_DIR "./output/jede_R_50_FPN/test_1"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
#        DATASETS.TEST_VIDEO_IDS [2] \
#        OUTPUT_DIR "./output/jede_R_50_FPN/test_2"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
#        DATASETS.TEST_VIDEO_IDS [3] \
#        OUTPUT_DIR "./output/jede_R_50_FPN/test_3"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
#        DATASETS.TEST_VIDEO_IDS [4] \
#        OUTPUT_DIR "./output/jede_R_50_FPN/test_4"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [4] \
#        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
#        INPUT.AUG.COPY_PASTE_MIX 0 \
#        OUTPUT_DIR "./output/jede_R_50_FPN/test_5"

# actually better results at this epoch

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        MODEL.WEIGHTS output/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8/model_0019999.pth

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --resume \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        OUTPUT_DIR "./output/jede_R_50_FPN/test_1"
#        MODEL.WEIGHTS "./output/jede_R_50_FPN/test_1/model_0019999.pth"

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --resume \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [2] \
        OUTPUT_DIR "./output/jede_R_50_FPN/test_2"

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --resume \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [3] \
        OUTPUT_DIR "./output/jede_R_50_FPN/test_3"

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --resume \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [4] \
        INPUT.MAX_SIZE_TEST 1600 \
        INPUT.MIN_SIZE_TEST 960 \
        OUTPUT_DIR "./output/jede_R_50_FPN/test_4"

# eval
#python train_net.py \
#        --resume \
#        --eval-only \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [4] \
#        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
#        INPUT.AUG.COPY_PASTE_MIX 0 \
#        INPUT.MAX_SIZE_TEST 400 \
#        INPUT.MIN_SIZE_TEST 240 \
#        OUTPUT_DIR "./output/jede_R_50_FPN/test_5"