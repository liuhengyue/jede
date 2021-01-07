cd ..
# train, cross validation on four videos
#python train_net.py \
#        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x_test_0.yaml \
#        --num-gpus 2

python train_net.py \
        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x_test_1.yaml \
        --num-gpus 2

python train_net.py \
        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x_test_2.yaml \
        --num-gpus 2

#python train_net.py \
#        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x_test_3.yaml \
#        --num-gpus 2



# test
#python train_net.py \
#        --config-file configs/faster_rcnn/faster_rcnn_R_50_FPN_1x.yaml \
#        --eval-only \
#        --resume \
#        --num-gpus 2