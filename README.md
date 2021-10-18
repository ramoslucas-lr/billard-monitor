# Billard Monitor

The Billard Monitor is a project developed as part of the Digital Image Processing course at the Federal University of Santa Catarina [UFSC](http://ufsc.br/). You can check the video with every step [here](https://www.youtube.com/watch?v=MnL_hqEmVZA).

## Requirements
```
opencv-python~=4.5.1.48
scipy~=1.6.3
pandas~=1.2.4
scikit-learn~=0.24.2
matplotlib~=3.4.1
numpy~=1.20.2
joblib~=1.0.1
```
All the requirements are available at the requirements.txt file and you can install it using 
```
 pip install -r requirements.txt
```

## Usage
No arguments are being passed to the code. To run the code you need to alter the variables _render_output_ and _build_model_ to decide if you want to render the output to a .avi file and build a RandomForest model, respectively. You need to have a built model to classify each circle detected by the Hough Transform.
```
python main.py
```
