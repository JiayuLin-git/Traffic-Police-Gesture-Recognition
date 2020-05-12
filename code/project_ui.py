import tkinter as tk
from tkinter.filedialog import askopenfilename
#import gesture_process
import os
import re
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#import Image

def NN():
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    data_points = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/data.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                data_points.append(sample)

    data_points_xycl = np.array(data_points)
    data_points_xyc = data_points_xycl[:, :-1]
    y = data_points_xycl[:, -1]

    # centralize datapoints and normalize
    data_points_xy_cent = []
    for row in data_points_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        data_points_xy_cent.append(new_row)

    result_point = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                result_point.append(sample)

    result_point_xycl = np.array(result_point)
    result_point_xyc = result_point_xycl[:, :-1]
    result_point_y = result_point_xycl[:, -1]

    result_point_xy_cent = []
    for row in result_point_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        result_point_xy_cent.append(new_row)




    '''sum = 0
    gesture_results = []
    for i in range(100):
        data_points_xy_train, data_points_xy_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.3)
        clf = MLPClassifier(hidden_layer_sizes=(512,))
        clf.fit(data_points_xy_train, y_train)
        gesture_results.append(clf.predict([result_point_xy_cent[0]])[0])
        score = clf.score(data_points_xy_test, y_test)
        #print(score)
        sum += score'''

    X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
    scaler = preprocessing.StandardScaler().fit(X_train)
    #print(scaler.mean_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    r_X_scaled = scaler.transform(result_point_xy_cent)

    sum = 0
    clf = MLPClassifier(hidden_layer_sizes=(1024,1024),solver='lbfgs')
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled,y_train)
        #print(clf.n_iter_, end=" ")=
        score_train = clf.score(X_train_scaled, y_train)
        #print(score_train, end=" ")
        score_test = clf.score(X_test_scaled, y_test)
        sum += score_test #print(score_test)

    tf = (clf.predict([r_X_scaled[0]])[0] == result_point_y[0])

    return clf.predict([r_X_scaled[0]])[0], sum / 10, tf

def HGB():
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    data_points = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/data.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                data_points.append(sample)

    data_points_xycl = np.array(data_points)
    data_points_xyc = data_points_xycl[:, :-1]
    y = data_points_xycl[:, -1]

    # centralize datapoints and normalize
    data_points_xy_cent = []
    for row in data_points_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        data_points_xy_cent.append(new_row)

    result_point = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                result_point.append(sample)

    result_point_xycl = np.array(result_point)
    result_point_xyc = result_point_xycl[:, :-1]
    result_point_y = result_point_xycl[:, -1]

    result_point_xy_cent = []
    for row in result_point_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        result_point_xy_cent.append(new_row)




    '''sum = 0
    gesture_results = []
    for i in range(100):
        data_points_xy_train, data_points_xy_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.3)
        clf = MLPClassifier(hidden_layer_sizes=(512,))
        clf.fit(data_points_xy_train, y_train)
        gesture_results.append(clf.predict([result_point_xy_cent[0]])[0])
        score = clf.score(data_points_xy_test, y_test)
        #print(score)
        sum += score'''

    X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
    scaler = preprocessing.StandardScaler().fit(X_train)
    #print(scaler.mean_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    r_X_scaled = scaler.transform(result_point_xy_cent)

    sum = 0
    clf = HistGradientBoostingClassifier()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled,y_train)
        #print(clf.n_iter_, end=" ")=
        score_train = clf.score(X_train_scaled, y_train)
        #print(score_train, end=" ")
        score_test = clf.score(X_test_scaled, y_test)
        sum += score_test #print(score_test)

    tf = (clf.predict([r_X_scaled[0]])[0] == result_point_y[0])

    return clf.predict([r_X_scaled[0]])[0], sum / 10, tf

def RF():
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    data_points = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/data.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                data_points.append(sample)

    data_points_xycl = np.array(data_points)
    data_points_xyc = data_points_xycl[:, :-1]
    y = data_points_xycl[:, -1]

    # centralize datapoints and normalize
    data_points_xy_cent = []
    for row in data_points_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        data_points_xy_cent.append(new_row)

    result_point = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                result_point.append(sample)

    result_point_xycl = np.array(result_point)
    result_point_xyc = result_point_xycl[:, :-1]
    result_point_y = result_point_xycl[:, -1]

    result_point_xy_cent = []
    for row in result_point_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        result_point_xy_cent.append(new_row)

    X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
    scaler = preprocessing.StandardScaler().fit(X_train)
    #print(scaler.mean_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    r_X_scaled = scaler.transform(result_point_xy_cent)

    sum = 0
    clf = RandomForestClassifier()
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled,y_train)
        #print(clf.n_iter_, end=" ")=
        score_train = clf.score(X_train_scaled, y_train)
        #print(score_train, end=" ")
        score_test = clf.score(X_test_scaled, y_test)
        sum += score_test #print(score_test)

    tf = (clf.predict([r_X_scaled[0]])[0] == result_point_y[0])

    return clf.predict([r_X_scaled[0]])[0], sum / 10, tf

def SVC():
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    data_points = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/data.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                data_points.append(sample)

    data_points_xycl = np.array(data_points)
    data_points_xyc = data_points_xycl[:, :-1]
    y = data_points_xycl[:, -1]

    # centralize datapoints and normalize
    data_points_xy_cent = []
    for row in data_points_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        data_points_xy_cent.append(new_row)

    result_point = []
    with open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.csv', 'r') as fd:
        for row in fd:
            row_list = row[:-1].split(',')
            sample = [float(i) for i in row_list[:-1]]
            sample.append(actions.index(row_list[-1]))
            if len(sample) == 76:
                result_point.append(sample)

    result_point_xycl = np.array(result_point)
    result_point_xyc = result_point_xycl[:, :-1]
    result_point_y = result_point_xycl[:, -1]

    result_point_xy_cent = []
    for row in result_point_xyc:
        # print(row)
        avg_x = row[3]
        avg_y = row[4]
        head_length = ((row[0] - row[3])**2 + (row[1] - row[4])**2)**0.5
        shoulder_length = ((row[3] - row[6])**2 + (row[4] - row[7])**2)**0.5
        new_row = []
        for i in range(16): # first 16 points
            new_row.append((row[3*i] - avg_x)/shoulder_length)
            new_row.append((row[3*i+1] - avg_y)/head_length)
            new_row.append(row[3*i+2]) # conf

        result_point_xy_cent.append(new_row)

    X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
    scaler = preprocessing.StandardScaler().fit(X_train)
    #print(scaler.mean_)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    r_X_scaled = scaler.transform(result_point_xy_cent)

    sum = 0
    clf = NuSVC(nu=0.2)
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data_points_xy_cent, y, test_size=0.4)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf.fit(X_train_scaled,y_train)
        #print(clf.n_iter_, end=" ")=
        score_train = clf.score(X_train_scaled, y_train)
        #print(score_train, end=" ")
        score_test = clf.score(X_test_scaled, y_test)
        sum += score_test #print(score_test)

    tf = (clf.predict([r_X_scaled[0]])[0] == result_point_y[0])

    return clf.predict([r_X_scaled[0]])[0], sum / 10, tf

def read_pic(txt_file, picture_path):
    picture_data = []
    with open(txt_file, 'r') as f:
        for row in f:
            l = re.findall(r"\d+\.?\d*", row)
            if len(l) == 6:
                a = float(l[0]) * pow(10, float(l[1]))
                b = float(l[2]) * pow(10, float(l[3]))
                c = float(l[4]) * pow(10, float(l[5]))
            if len(l) == 3:
                a = float(l[0])
                b = float(l[1])
                c = float(l[2])
            picture_data.append(a)
            picture_data.append(b)
            picture_data.append(c)

    idx = picture_path[80:].find('/')
    label = picture_path[80:][:idx]

    picture_data.append(label)

    with open("/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(picture_data)

def hit_RF():
    picture_path = askopenfilename()
    var_title.set('Recogintion Method: Random Forest')
    l_title.pack()
    #print(picture_path)
    choose_picture_RF(picture_path)

def choose_picture_RF(picture_path):
    os.system(command='cd /Users/zephyryau/openpose/build/examples/tutorial_api_python')
    os.system(command='python3 input_picture.py ' + picture_path)

    read_pic('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.txt', picture_path)

    print("RF")
    gesture_result, accuracy, tf = RF()
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    var1.set('Recogintion Result: ' + actions[int(gesture_result)])
    var2.set('Accuracy: ' + str(accuracy))
    if tf:
        var3.set('The result is correct')
    else:
        var3.set('The result is not correct')

    image = Image.open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    #print(image.size)
    new_image = image.resize((int(image.size[0]/image.size[1]*500), 500), Image.ANTIALIAS)
    image_file = ImageTk.PhotoImage(new_image)
    #image_file = tk.PhotoImage(file='/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    img = canvas.create_image(250, 0, anchor='n',image=image_file)
    img.pack()

def hit_NN():
    picture_path = askopenfilename()
    var_title.set('Recogintion Method: Neural Network')
    l_title.pack()
    #print(picture_path)
    choose_picture_NN(picture_path)

def choose_picture_NN(picture_path):
    os.system(command='cd /Users/zephyryau/openpose/build/examples/tutorial_api_python')
    os.system(command='python3 input_picture.py ' + picture_path)

    read_pic('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.txt', picture_path)

    gesture_result, accuracy, tf = NN()
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    var1.set('Recogintion Result: ' + actions[int(gesture_result)])
    var2.set('Accuracy: ' + str(accuracy))
    if tf:
        var3.set('The result is correct')
    else:
        var3.set('The result is not correct')

    image = Image.open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    #print(image.size)
    new_image = image.resize((int(image.size[0]/image.size[1]*500), 500), Image.ANTIALIAS)
    image_file = ImageTk.PhotoImage(new_image)
    #image_file = tk.PhotoImage(file='/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    img = canvas.create_image(250, 0, anchor='n',image=image_file)
    img.pack()

def hit_SVC():
    picture_path = askopenfilename()
    var_title.set('Recogintion Method: SVM')
    l_title.pack()
    #print(picture_path)
    choose_picture_SVC(picture_path)

def choose_picture_SVC(picture_path):
    os.system(command='cd /Users/zephyryau/openpose/build/examples/tutorial_api_python')
    os.system(command='python3 input_picture.py ' + picture_path)
    
    read_pic('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.txt', picture_path)

    gesture_result, accuracy, tf = SVC()
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    var1.set('Recogintion Result: ' + actions[int(gesture_result)])
    var2.set('Accuracy: ' + str(accuracy))
    if tf:
        var3.set('The result is correct')
    else:
        var3.set('The result is not correct')

    image = Image.open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    #print(image.size)
    new_image = image.resize((int(image.size[0]/image.size[1]*500), 500), Image.ANTIALIAS)
    image_file = ImageTk.PhotoImage(new_image)
    #image_file = tk.PhotoImage(file='/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    img = canvas.create_image(250, 0, anchor='n',image=image_file)
    img.pack()

def hit_HGB():
    picture_path = askopenfilename()
    var_title.set('Recogintion Method: Histogram-based Gradient Boosting')
    l_title.pack()
    #print(picture_path)
    choose_picture_HGB(picture_path)

def choose_picture_HGB(picture_path):
    os.system(command='cd /Users/zephyryau/openpose/build/examples/tutorial_api_python')
    os.system(command='python3 input_picture.py ' + picture_path)

    read_pic('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.txt', picture_path)

    gesture_result, accuracy, tf = HGB()
    actions = ['change_lane', 'pull_over', 'slow', 'stop', 'straight', 'turn_left', 'turn_right', 'wait_to_turn_left']

    var1.set('Recogintion Result: ' + actions[int(gesture_result)])
    var2.set('Accuracy: ' + str(accuracy))
    if tf:
        var3.set('The result is correct')
    else:
        var3.set('The result is not correct')

    image = Image.open('/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    #print(image.size)
    new_image = image.resize((int(image.size[0]/image.size[1]*500), 500), Image.ANTIALIAS)
    image_file = ImageTk.PhotoImage(new_image)
    #image_file = tk.PhotoImage(file='/Users/zephyryau/Documents/study/INF552/Project/input_picture/result.png')
    img = canvas.create_image(250, 0, anchor='n',image=image_file)
    img.pack()

window = tk.Tk()
window.title('INF552 Project: Traffic Police Gesture Recognition')
window.geometry('500x800')

var_title = tk.StringVar()
var1 = tk.StringVar()
var2 = tk.StringVar()
var3 = tk.StringVar()
l_title = tk.Label(window, textvariable=var_title, bg='gray', fg='white', font=('Arial', 24), width=100, height=2)
l1 = tk.Label(window, textvariable=var1, bg='gray', fg='white', font=('Arial', 24), width=100, height=2)
l2 = tk.Label(window, textvariable=var2, bg='gray', fg='white', font=('Arial', 24), width=100, height=2)
l3 = tk.Label(window, textvariable=var3, bg='gray', fg='white', font=('Arial', 24), width=100, height=2)
l_title.pack()
l1.pack()
l2.pack()
l3.pack()

canvas = tk.Canvas(window, bg='gray', height=400, width=500)
canvas.pack()

b_RF  = tk.Button(window, text='Choose a picture and recoginze through Random Forest', command=hit_RF)
b_NN = tk.Button(window, text='Choose a picture and recoginze through Neural Network', command=hit_NN)
b_SVC = tk.Button(window, text='Choose a picture and recoginze through SVM', command=hit_SVC)
b_HGB = tk.Button(window, text='Choose a picture and recoginze through Histogram-based Gradient Boosting', command=hit_HGB)

b_RF.pack()
b_NN.pack()
b_SVC.pack()
b_HGB.pack()

window.mainloop()
