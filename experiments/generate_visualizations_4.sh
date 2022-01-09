# JEDE baseline

python train_net.py \
        --resume \
        --eval-only \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TEST_VIDEO_IDS [4] \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        INPUT.MAX_SIZE_TEST 1920 \
        INPUT.MIN_SIZE_TEST 1280 \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_4_imagesizex4"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_4_imagesizex4"

# JEDE augmented

python train_net.py \
        --resume \
        --eval-only \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml \
        DATASETS.TEST_VIDEO_IDS [4] \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        OUTPUT_DIR "./output/jede_R_50_FPN/test_4_imagesizex4"
#                INPUT.MAX_SIZE_TEST 1920 \
#        INPUT.MIN_SIZE_TEST 1280 \

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/jede_R_50_FPN/test_4_imagesizex4"



# faster rcnn

python train_net.py \
        --resume \
        --eval-only \
        --num-gpus 2 \
        --config-file configs/faster_rcnn/test_4.yaml \
        DATASETS.TEST_VIDEO_IDS [4] \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        INPUT.MAX_SIZE_TEST 1920 \
        INPUT.MIN_SIZE_TEST 1280 \
        OUTPUT_DIR "./output/faster_rcnn/test_4_imagesizex4"

python visualize_json_results.py --config-file configs/faster_rcnn/test_4.yaml OUTPUT_DIR "./output/faster_rcnn/test_4_imagesizex4"


# cascade rcnn

python train_net.py \
        --resume \
        --eval-only \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TEST_VIDEO_IDS [4] \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        INPUT.MAX_SIZE_TEST 1920 \
        INPUT.MIN_SIZE_TEST 1280 \
        OUTPUT_DIR "./output/cascade_rcnn/test_4_imagesizex4"

python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/cascade_rcnn/test_4_imagesizex4"

# tridentnet

python train_net.py \
        --resume \
        --eval-only \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TEST_VIDEO_IDS [4] \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        INPUT.MAX_SIZE_TEST 1920 \
        INPUT.MIN_SIZE_TEST 1280 \
        OUTPUT_DIR "./output/trident_net/test_4_imagesizex4"

python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/trident_net/test_4_imagesizex4"
