# Dataset Download Instructions

If you are not using the pre-stored datasets, you can download VOC2007, VOC2012, and COCO datasets using the instructions below.

## PASCAL VOC 2007
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
```

## PASCAL VOC 2012
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

tar xvf VOCtrainval_11-May-2012.tar
```

## COCO
For unsupervised object discovery, the standard practice often uses the COCO 2014 dataset.
```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
```
