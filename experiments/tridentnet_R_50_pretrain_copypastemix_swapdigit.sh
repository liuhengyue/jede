# train, cross validation on four videos
python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
        DATASETS.TEST_VIDEO_IDS [0] \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_0"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,2,3] \
        DATASETS.TEST_VIDEO_IDS [1] \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_1"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,3] \
        DATASETS.TEST_VIDEO_IDS [2] \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_2"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2] \
        DATASETS.TEST_VIDEO_IDS [3] \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_3"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] \
        DATASETS.TEST_VIDEO_IDS [4] \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_4"

python train_net.py \
        --num-gpus 2 \
        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
        DATASETS.TRAIN_VIDEO_IDS [4] \
        DATASETS.TEST_VIDEO_IDS [0,1,2,3] \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_5"


# eval
#python train_net.py --eval-only --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml MODEL.WEIGHTS "./output/faster_rcnn/test_0/model_0019999.pth"
#python train_net.py --eval-only --num-gpus 2 --config-file configs/faster_rcnn/test_1.yaml MODEL.WEIGHTS MODEL.WEIGHTS "./output/faster_rcnn/test_1/model_0019999.pth"

#python train_net.py --num-gpus 2 --eval-only --config-file configs/faster_rcnn/test_0.yaml \
#          MODEL.WEIGHTS "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_0/model_0019999.pth" \
#          INPUT.AUG.COPY_PASTE_MIX 5 \
#          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
#          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_0"


#python train_net.py \
#        --num-gpus 2 \
#        --resume \
#        --eval-only \
#        --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml \
#        DATASETS.TRAIN_VIDEO_IDS [1,2,3] \
#        DATASETS.TEST_VIDEO_IDS [0] \
#        OUTPUT_DIR "./output/trident_net/pretrain_copypastemix_swapdigit/test_0"