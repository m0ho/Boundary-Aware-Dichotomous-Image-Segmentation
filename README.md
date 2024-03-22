## Usage
### Requirements
* Python 3.8
* Pytorch 1.7.1
* OpenCV
* Numpy
* Apex
* Timm

### Directory

````
.
|-- README.md
|-- model
|   |-- model-56.pth
|-- pre
|   |-- resnet18.pth
|   `-- swin224.pth
|-- result
`-- src
    |-- PGNet.py
    |-- Res.py
    |-- ResNet_models_Custom.py
    |-- Swin.py
    |-- __pycache__
    |   |-- Res.cpython-310.pyc
    |   |-- ResNet_models_Custom.cpython-310.pyc
    |   `-- Swin.cpython-310.pyc
    |-- dataset.py
    |-- edge_prediction.py
    |-- test.py
    |-- test.sh
    |-- train.py
    |-- train.sh
    `-- utils
        |-- __init__.py
        `-- lr_scheduler.py
````

### Train
```
cd src
./train.sh
```

### Test
The trained model can be download here: [Google Drive](https://drive.google.com/drive/folders/1hXwCvrdmvkaRePXWPTw5tjFXmrrzHPtt?usp=sharing)

```
cd src
./test.sh
```
* After testing, saliency maps will be saved in RESULT folder



## Citation
```
@article{tang2024boundary,
  title={Boundary-aware dichotomous image segmentation},
  author={Tang, Haonan and Chen, Shuhan and Liu, Yang and Wang, Shiyu and Chen, Zeyu and Hu, Xuelong},
  journal={The Visual Computer},
  pages={1--12},
  year={2024},
  publisher={Springer}
}
```
