python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        OUTPUT_DIR "./output/ablations/augmentation/swapdigit"


python train_net.py \
        --num-gpus 2 \
        --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        MODEL.ROI_NECK_BASE_BRANCHES.NORM "" \
        MODEL.ROI_DIGIT_NECK_OUTPUT.NORM "" \
        MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
        INPUT.AUG.COPY_PASTE_MIX 5 \
        INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
        OUTPUT_DIR "./output/ablations/augmentation/pretrain_copypastemix_swapdigit"

#python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml \
#          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
#          OUTPUT_DIR "./output/faster_rcnn/swapdigit/test_0"