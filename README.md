# Car Counting on Highway Using Customized YOLOv8 Object Counter
This project aims to count cars entering or exiting a specified line on the highway. It includes object tracking and object counting functionalities.<br>

The basic functionality is derived from the YOLO object counting example. However, the object counter has been customized to accurately count cars entering or exiting a specific region on the highway.<br>

This approach can be adapted to count various items in different contexts and applications.<br>

## Installation

This package is tested on Ubuntu 20.04 with Python 3.9.12. First, create your virtual environment:

```shell
python -m venv venv
source venv/bin/activate
```
Next, install all dependencies:

```shell
pip install -r requirements.txt
```

You can pass the desired paths for inputs, outputs and models directory as arguments in CLI. the default configuration is considering these directories as follows in the root of the project: <br>

```
├── inputs
├── models
├── outputs
│   └── output.avi
```
You can customize the following arguments:

* -o or --outputs: path to output video files. default=`outputs`.
* -i or --inputs: path to input video files. default=`inputs`.
* -m or --models: path to dir in which all of saved models are stored. default=`models`.
* -on or --output_name: name of the output video with annotated bounding boxes and counting results on it. default=`output.avi`.
* -in or --input_name: name of the input video.default=None.
* -mn or --model_name: name of the saved model. default=`yolov8n.pt`.

## Inference with Trained Models
To get predictions from a YOLO saved model, run:

```shell
python count.py --input_name 'path/to/test_video.mp4'
```