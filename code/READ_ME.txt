(1) gesture_process.py
function: Through the OpenPose module, process the original data and output the human skeleton diagram and body points' coordinates.
Input: traffic_police_gesture_original
Output: traffic_police_gesture_processed

(2) data_process.py
function: Integrate the coordinates of the body points of all pictures into an Excel file.
Input: traffic_police_gesture_processed
Output: data.csv

(3) input_picture.py
function: Through the OpenPose module, process the original data and output the human skeleton diagram and body points' coordinates.
Input: Picture path obtained through the choose-file button of the Pythonn GUI
Output: result.png / result.txt

(4) project_ui.py
function: Build a Python GUI. This GUI allows the user to choose picture and get the gesture recognition through four models: Neural Network, Random Forest, SVM, and Hist Gradient Boosting
Input: choose-file button
Output: The visualization of the recognition result, the accuracy of models, and Whether the result is correct or not

To specific, OpenPose module is supposed to install in advance to run (1)(3)(4).



(5) training_202005100342.ipynb
function: Build four models: Neural Network, Random Forest, SVM, and Histogram-based Gradient Boosting and evaluate their accuracy.
Input: data.csv
Output: scaler/Boosting/MLP/RF/SVM (the files to store models' information)

(6) postprocess.ipynb
Function: Evaluate the effect of the model
Input: data.csv
Output: confusion matrix of four models
