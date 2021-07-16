# Pose-guided R-CNN

## Installation

### Dependencies

Tested on PyTorch 1.8.1.

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install opencv-python
```



### Our code base and Detectron2

```
git clone https://github.com/liuhengyue/pgrcnn.git
cd pgrcnn
git submodule init
git submodule update
python -m pip install -e detectron2
```

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html) for more details on installing detectron2.


## Dataset Preparation

### Prepare Jersey Number

```
mkdir datasets/jnw
```

### Prepare COCO (Optional)
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir datasets/coco
unzip train2017.zip -d datasets/coco
unzip val2017.zip -d datasets/coco
unzip annotations_trainval2017.zip -d datasets/coco

rm train2017.zip val2017.zip annotations_trainval2017.zip
```
### Prepare SVHN (Optional)
```
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
mkdir svhn
tar -xvzf train.tar.gz -C svhn
rm train.tar.gz
```