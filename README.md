
Train a YOLOv5 model on a custom dataset.
datasets download from lz620abcd@163.com.

**0.Setup**
------
* git clone https://github.com/lz620abc/yolo_detection.git
* cd yolo_detection
* pip install -qr requirements.txt

**1.Detect**
------
* python detect.py --weights '' --conf 0.25 --source data/images
  
![image](https://github.com/lz620abc/yolo_detection/blob/main/Results/Detection_results.png)

**2.Train**
------
* Usage - Single-GPU training:
     $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

* Usage - Multi-GPU DDP training:
     $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights '' --img 640 --device 0,1,2,3

**3.Precision Result**
--------

![image](https://github.com/lz620abc/yolo_detection/blob/main/Results/Precison.jpg)
  
