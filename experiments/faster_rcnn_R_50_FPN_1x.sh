# train, cross validation on four videos
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_1.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_2.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_3.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_4.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_5.yaml

# eval
python train_net.py --eval-only --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml MODEL.WEIGHTS "./output/faster_rcnn/test_0/model_0019999.pth"
python train_net.py --eval-only --num-gpus 2 --config-file configs/faster_rcnn/test_1.yaml MODEL.WEIGHTS "./output/faster_rcnn/test_1/model_0019999.pth"

python train_net.py --num-gpus 2 --eval-only --resume --config-file configs/faster_rcnn/test_5.yaml INPUT.MAX_SIZE_TEST 400 INPUT.MIN_SIZE_TEST 240