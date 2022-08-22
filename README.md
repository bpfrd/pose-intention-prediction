

# pose & bounding box prediction with intention

An open source code to predict human pose and bounding box as well as intention in JAAD dataset. 

## Contents
------------
  * [Repository Structure](#repository-structure)
  * [Proposed Method](#proposed-method)
  * [Results](#results)
  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Training/Testing](#training-testing)
  
## Repository structure:
------------
    ├── bbox-prediction-in-JAAD-with-intention         : bounding box & intention project repository
            ├── train.py                : Script for training including all necessary functions such as visualization and network  
            ├── test.py                 : Script for testing.  
            ├── DataLoader.py           : Script for data pre-processing and loader. 
    
    ├── pose-prediction-in-JAAD-with-intention         : pose & intention project repository
            ├── train.py                : Script for training including all necessary functions such as visualization and network  
            ├── test.py                 : Script for testing.  
            ├── DataLoader.py           : Script for data pre-processing and loader. 
            
## Proposed method
-------------
We used a simple LSTM network similar to [Pedestrian Intention Prediction: A Multi-task Perspective
](https://arxiv.org/abs/2010.10270) to predict both pose and bounding boxes as well as intention. 


## Results
--------------
In the below visualizations, we have shown observed frames in blue, ground truth future frames in green, and the predicted future frames in red. In addition, C and NC represent human crossing and non-crossing respectively.

blue: observation frames

red: predicted future frames

green: ground truth future frames 

While our model can very accurately predict bounding boxes and intentions, it doesn't perform very well on the prediction of poses. 

![Example of outputs](bbox-prediction-in-JAAD-with-intention/visualization.gif)
![Example of outputs](pose-prediction-in-JAAD-with-intention/visualization.gif)
  
## Installation:
------------
Start by cloning this repositiory:
```
git clone https://github.com/vita-epfl/pose-intention.git
cd pose-intention
```
Create and activate virtual environment:
```
pip install --upgrade virtualenv
virtualenv -p python3 <venvname>  
source <venvname>/bin/activate  
pip install --upgrade pip
```
And install the dependencies:
```
pip install -r requirements.txt
```

## Dataset:
  
  We provided a preprocessed and balanced subset of JAAD dataset in each project's directory so that it is not needed to download and preprocess the original JAAD dataset. 
  
  
## Training/Testing:
Open `train.py` and `test.py` and change the parameters in the args class depending on the paths of your files.
Start training the network by running the command:
```
python3 train.py
```
Test the trained network by running the command:
```
python3 test.py
```



### Citation

```
@inproceedings{bouhsain2020pedestrian,
title={Pedestrian Intention Prediction: A Multi-task Perspective},
 author={Bouhsain, Smail and Saadatnejad, Saeed and Alahi, Alexandre},
  booktitle = {European Association for Research in Transportation  (hEART)},
  year={2020},
}
```
