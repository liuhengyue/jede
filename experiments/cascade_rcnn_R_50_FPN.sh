python train_net.py \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
        DATASETS.TEST_VIDEO_IDS [0] \
        OUTPUT_DIR "./output/cascade_rcnn/test_0"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        OUTPUT_DIR "./output/cascade_rcnn/test_1"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
        DATASETS.TEST_VIDEO_IDS [2] \
        OUTPUT_DIR "./output/cascade_rcnn/test_2"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
        DATASETS.TEST_VIDEO_IDS [3] \
        OUTPUT_DIR "./output/cascade_rcnn/test_3"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        DATASETS.TEST_VIDEO_IDS [4] \
        OUTPUT_DIR "./output/cascade_rcnn/test_4"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        OUTPUT_DIR "./output/cascade_rcnn/test_5"

python train_net.py \
        --num-gpus 2 \
        --resume \
        --eval-only \
        --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        OUTPUT_DIR "./output/cascade_rcnn/test_5" \
        INPUT.MAX_SIZE_TEST 400 \
        INPUT.MIN_SIZE_TEST 240