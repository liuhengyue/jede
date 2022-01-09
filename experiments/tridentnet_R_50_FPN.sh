# train, cross validation on four videos
python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
        DATASETS.TEST_VIDEO_IDS [0] \
        OUTPUT_DIR "./output/trident_net/test_0"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        OUTPUT_DIR "./output/trident_net/test_1"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
        DATASETS.TEST_VIDEO_IDS [2] \
        OUTPUT_DIR "./output/trident_net/test_2"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
        DATASETS.TEST_VIDEO_IDS [3] \
        OUTPUT_DIR "./output/trident_net/test_3"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        DATASETS.TEST_VIDEO_IDS [4] \
        OUTPUT_DIR "./output/trident_net/test_4"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        OUTPUT_DIR "./output/trident_net/test_5"

# eval

python train_net.py \
        --num-gpus 2 \
        --resume \
        --eval-only \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        OUTPUT_DIR "./output/trident_net/test_5" \
        INPUT.MAX_SIZE_TEST 400 \
        INPUT.MIN_SIZE_TEST 240

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
        DATASETS.TEST_VIDEO_IDS [0] \
        OUTPUT_DIR "./output/trident_net/test_0" \
        MODEL.WEIGHTS "./output/trident_net/test_0/model_0019999.pth"

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        OUTPUT_DIR "./output/trident_net/test_1" \
        MODEL.WEIGHTS "./output/trident_net/test_1/model_0019999.pth"