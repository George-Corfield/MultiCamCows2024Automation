# MultiCamCows2024Automation
An Automated Pipeline for training and inference on the MultiCamCows2024 dataset

## Initial Setup
Run the following in your virtual environment:

```pip install -r requirements.txt```

## Running for Training

It is recommended that the video directory and save directory are saved under the project file. They should be organised in the following way with the save direcoty being empty initially for training

### File System Structure

```
project/
├── videos/
|   ├── day1/
|   │   ├── camera-1/
|   │   │   ├── video.mp4
|   │   │   └── video.avi
|   │   └── camera-2/
|   │       ├── video.mp4
|   │       └── video.avi
|   ├── day2/
|   │   ├── camera-1/
|   │   |   ├── video.mp4
|   │   |   └── video.avi
|   |   └── camera-2/
|   │       ├── video.mp4
|   │       └── video.avi
...  ...
├── save/
```

Note that this project will produce a large set of images from the given videos stored in the save directory. In general 30,000 images equates to 2.5Gb.

To run the code for training follow the format below for terminal:

```

python training.py train path/to/video/directory path/to/save/directory

```

Where:
- "train" initiates the training procedure
- the path to the video directory in the format of "project/videos/" for the above directory structre
- the path to the save direcotyr in the format of "project/save/" for the above directory structure

This training procedure produces a set of images under the save directory that mimic the video file structure. It also creates a models folder that contains both the classification model as well as the embedding model. On top of this there is a lightning logs file that can be used for training optimisation.

It is important to note that this training procedure is set up to be run on Blue Crystal Phase 4 HPC. Elements of the files ```train-simclr.py```, ```simCLR.py``` and ```cluster.py``` should be changed to fit the training environment. It is also important to note that the classifier is set up as a pickle file, which requires it to be re-run in the same environment for downstream classification.

## Running for inference tasks

For inference tasks, there first must exist an embedding model and classification model. Images are not required and can be deleted to save space if necessary.

To run for inference run the following:

```

python training.py infer path/to/video/directory path/to/save/directory

```

Where the video directory is an input directory which follows the same structure as the training video directory, and the save directory contains your models directory. This is also where your output videos are stored.

The output of this file is a set of avi files for each camera and date input, which contain bounding boxes detecting and classifying each cow.

please email gacorfield50@gmail.com for more information or questions.
