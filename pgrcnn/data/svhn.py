from detectron2.data.datasets import register_coco_instances
register_coco_instances("svhn_train", {}, "../../../../datasets/jnw/train/train.json", "../../../../datasets/jnw/train")