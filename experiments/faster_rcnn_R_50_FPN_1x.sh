# train, cross validation on four videos
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_0.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_1.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_2.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_3.yaml
python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_4.yaml