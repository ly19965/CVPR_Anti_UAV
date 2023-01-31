# Prepare conda enviroment and dataset
## Create and activate conda environment
```
conda create -n anti_uav python=3.7 
conda activate anti_uav
```
## Intall dependency packages
```
cd /path/to/baseline/yolov5     
pip install -r requirements.txt
```
## Download the test dataset from anti-uav 3rd challenge (track1 or track2)

Set the dataset path (line 23) in test_track1.py and test_track2.py

# Test track1 tracker with initial target ground-truth box

## Test
```
cd /path/to/baseline
python test_track1.py
```

# Test track2 tracker and Yolov5 without initial target ground-truth box
Detect the initial state of target using yolov5 then track the target using SiamFC
(You can also use detectors and trackers together for localization.)

## Test
Set paths: set Yolov5 weight path (line 114-115) in the /path/to/baseline/detection_siamfc.py 

```
cd /path/to/baseline
export PYTHONPATH=/path/to/baseline/yolov5:$PYTHONPATH
python test_track2.py
```