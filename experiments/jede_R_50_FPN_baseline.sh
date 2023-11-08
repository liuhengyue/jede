# more will cause overfitting
#  STEPS: ()
#  MAX_ITER: 20000
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
#        DATASETS.TEST_VIDEO_IDS [0] \
#        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_0"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
#        DATASETS.TEST_VIDEO_IDS [1] \
#        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_1"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
#        DATASETS.TEST_VIDEO_IDS [2] \
#        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_2"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
#        DATASETS.TEST_VIDEO_IDS [3] \
#        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_3"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
#        DATASETS.TEST_VIDEO_IDS [4] \
#        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_4"
#
#python train_net.py \
#        --num-gpus 2 \
#        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [4] \
#        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
#        INPUT.AUG.COPY_PASTE_MIX 0 \
#        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_5"

# eval
python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
        DATASETS.TEST_VIDEO_IDS [0] \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_0" \
        MODEL.WEIGHTS "./output/jede_R_50_FPN_baseline/test_0/model_0019999.pth"

python train_net.py \
        --num-gpus 2 \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_1" \
        MODEL.WEIGHTS "./output/jede_R_50_FPN_baseline/test_1/model_0019999.pth"


python train_net.py \
        --num-gpus 2 \
        --resume \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
        DATASETS.TEST_VIDEO_IDS [2] \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_2"

python train_net.py \
        --num-gpus 2 \
        --resume \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
        DATASETS.TEST_VIDEO_IDS [3] \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_3"

python train_net.py \
        --num-gpus 2 \
        --resume \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        DATASETS.TEST_VIDEO_IDS [4] \
        INPUT.MAX_SIZE_TEST 1600 \
        INPUT.MIN_SIZE_TEST 960 \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_4"

python train_net.py \
        --num-gpus 2 \
        --resume \
        --eval-only \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        INPUT.AUG.COPY_PASTE_MIX 0 \
        OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_5" \
        INPUT.MAX_SIZE_TEST 400 \
        INPUT.MIN_SIZE_TEST 240