# train, cross validation on four videos
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml \
          MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_0"

python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_1.yaml \
          MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_1"

python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_2.yaml \
          MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_2"

python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_3.yaml \
          MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_3"

python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_4.yaml \
          MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_4"

python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_5.yaml \
          MODEL.WEIGHTS "output/pg_rcnn/aug/datasets_mix/pretrain_coco_svhn_3x/model_final.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_5"

# eval
#python train_net.py --eval-only --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml MODEL.WEIGHTS "./output/faster_rcnn/test_0/model_0019999.pth"
#python train_net.py --eval-only --num-gpus 2 --config-file configs/faster_rcnn/test_1.yaml MODEL.WEIGHTS MODEL.WEIGHTS "./output/faster_rcnn/test_1/model_0019999.pth"

python train_net.py --num-gpus 2 --eval-only --config-file configs/faster_rcnn/test_0.yaml \
          MODEL.WEIGHTS "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_0/model_0019999.pth" \
          INPUT.AUG.COPY_PASTE_MIX 5 \
          INPUT.AUG.HELPER_DATASET_NAME "svhn_train" \
          OUTPUT_DIR "./output/faster_rcnn/pretrain_copypastemix_swapdigit/test_0"
