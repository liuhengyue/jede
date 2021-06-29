# Pose-guided R-CNN

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html) for installing detectron2.


## Dataset Preparation

```
# prepare COCO
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir datasets/coco
unzip train2017.zip -d datasets/coco
unzip val2017.zip -d datasets/coco
unzip annotations_trainval2017.zip -d datasets/coco

rm train2017.zip val2017.zip annotations_trainval2017.zip

# prepare SVHN
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
mkdir svhn
tar -xvzf train.tar.gz -C svhn
rm train.tar.gz
```