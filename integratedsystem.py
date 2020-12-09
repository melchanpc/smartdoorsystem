#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 00:38:11 2020

"""

import Adafruit_DHT
from filestack import Client
import os
from pandas import read_csv
from matplotlib import pyplot
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import joblib
from multiprocessing import Process
import numpy as np
import codecs, json 
import requests
import time

def video_recorder(duration):
    
    os.system('raspivid -t 10000 -w 640 -h 480 -rot 180 -o pivideo.h264')
    os.system('MP4Box -add ' + 'pivideo.h264' + ' ' + 'pivideo.mp4')
#              print("uploading filestack...")
#              new_filelink = client.upload(filepath="/home/pi/pivideo.mp4") #path to you image
             # print("Posting to IFTTT...")
             # r = requests.post(
             # "https://maker.ifttt.com/trigger/trigger/with/key/bjiw0jg3AvS3HFEo9Kv5PV",
             # json={"value1" : new_filelink.url}) #one line # ifttt api key = hjklyuioi
             # print("upload complete")
    
    return 0

#import board
#import adafruit_dht
def send_telegram(list_distance):
    start1 = time.time()
    start = time.time()
    duration1 = 0
    h =1
    send = []
    move = ['standstill','move further away','move nearer']
    move = np.array(move)
    dura_post = np.zeros((3))
    sensor=Adafruit_DHT.AM2302
    gpio = 18
    humidity, temperature = Adafruit_DHT.read(sensor, gpio)
                                       
    print('humidity: {}, temperature: {}' .format(humidity,temperature))
    thresh_h = 80
    thresh_l = 60
    threst_h = 40
    threst_l = 20
    summer = 0
    rain = 0
    normal = 0
    expect = "normal"
    while duration1 < 10:
#     print("collecting distance....")
        print("start detecting...")
        for j in range(number_distance):
            
            GPIO.output(PIN_TRIGGER,GPIO.LOW)
            time.sleep(0.2)
            
            GPIO.output(PIN_TRIGGER,GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(PIN_TRIGGER,GPIO.LOW)
            
            while GPIO.input(PIN_ECHO)==0:
                    pulse_start_time = time.time()
                    
            while GPIO.input(PIN_ECHO)==1:
                    pulse_end_time = time.time()
                    
               
            pulse_duration = pulse_end_time-pulse_start_time
            distance = round(17500*pulse_duration,2)
            
            #check for distance and record video
            
                
            list_distance[j] = distance 
        # make predictions
    #     print("finished")
        
        relative_distance1 = list_distance[0]-list_distance[1]
        relative_distance2 = list_distance[0]-list_distance[2]
    #     print(relative_distance1)
    #     print(relative_distance2)
        result = loaded_model.predict([[relative_distance1,relative_distance2]])
        result2 = loaded_model2.predict([[abs(relative_distance1),abs(relative_distance2)]])
        if h == 1:
            prevresult = result
            prevresult2 = result2
            h=0
        print("predicted movement: {}" .format(result))
        print("predicted speed of movement: {}" .format(result2))
        send.append([result[0],result2[0]])
        if (result != prevresult):
            end = time.time()
            duration = end-start
            start = time.time()
            print("detected person {} for {} seconds" .format(prevresult,duration))
            dura_post[np.where(move==prevresult[0])] += duration
            print(dura_post)
#             r = requests.post(
#                     "https://maker.ifttt.com/trigger/trigger/with/key/dWtemIT1rpLejVpcEw-MaC",
#                     json={"value1" : '{}' .format(prevresult[0]), "value2" : '{}' .format(prevresult2[0]),"value3" : duration })
#             
                         
        prevresult = result
        prevresult2 = result2
        end1 = time.time()
        duration1 = end1-start1
    send = np.array(send)
    standstill= send[np.where(send=='standstill')]
    further= send[np.where(send=='move further away')]
    near= send[np.where(send=='move nearer')]
    movements = ['stanstill','move further away','move nearer'] 
    detected_movement_index = np.argmax([len(standstill),len(further),len(near)])
    data = [len(standstill),len(further),len(near)] 
    detected_movement = str(movements[detected_movement_index])
# Creating plot
    
    fig,(ax1, ax2,ax3) = plt.subplots(1, 3)
#     fig1, ax1 = plt.subplots()
#     fig2, ax2 = plt.subplots()
#     fig3, ax3 = plt.subplots()
    ax1.pie(data, labels = movements) 
    
# show plot 
    
    #plt.savefig('foo.png')
    send2 = send[:,1]
    still= send2[np.where(send2=='still')]
    fast= send2[np.where(send2=='fast')]
    slow= send2[np.where(send2=='slow')]
    detected_speed_index = np.argmax([len(still),len(fast),len(slow)])
    speed = ['still','fast','slow'] 
    detected_speed = str(speed[detected_speed_index])
    data = [len(still),len(slow),len(fast)] 
  
# Creating plot 
  
    ax2.pie(data, labels = speed)
    
    ax3.bar(move,dura_post)
    ax3.set_xticklabels(move, rotation = 90)
    plt.savefig('foo.png')
#     r = requests.post(
#                     "https://maker.ifttt.com/trigger/tt/with/key/dWtemIT1rpLejVpcEw-MaC",
#                     json={"value1" : 'foo.png'})
            
    client = Client("AYHnX6THQBUt1z1uojUtQz")
    
    new_filelink = client.upload(filepath="pivideo.mp4") #path to you image
    os.system('rm pivideo.mp4')         
    r = requests.post("https://maker.ifttt.com/trigger/Video/with/key/dWtemIT1rpLejVpcEw-MaC",json={"value1" : new_filelink.url})

    new_filelink = client.upload(filepath="foo.png") #path to you image
             
    r = requests.post("https://maker.ifttt.com/trigger/tt/with/key/dWtemIT1rpLejVpcEw-MaC",json={"value1" : new_filelink.url})
# show plot
    #Detected humidity at the door
    high_h = 0
    low_h = 0
    if humidity > thresh_h:
        high_h = 1
        low_h = 0
    elif humidity < thresh_l:
        low_h = 1
        high_h = 0
    else:
        high_h = 0
        low_h = 0
    
    # Detected temperature at the door
    high_t = 0
    low_t = 0
    if temperature > threst_h:
        high_t = 1
        low_t = 0
    elif temperature < threst_l:
        low_t = 1
        high_t = 0
    else:
        high_t = 0
        low_t = 0
    if low_h and high_t:
        summer = 1
    elif high_h and low_t:
        rain = 1
    else:
        normal = 1
    # Provide more information to the movement
    if rain and (detected_speed == ('slow' or 'still')):
        expect = "Cold and humid...people may be looking for shaded area"
    elif summer and (detected_speed == ('slow' or 'still')):
        expect = "Hot and less humid...people may be looking for shaded area"
    elif summer and (detected_speed == 'fast'):
        expect = "Hot and humid...people are running around, may be because of fire" 
        
    r = requests.post("https://maker.ifttt.com/trigger/trigger/with/key/dWtemIT1rpLejVpcEw-MaC",json={"value1" : detected_movement,"value2": detected_speed, "value3": expect})

    
    return 0

#test trained model
data= read_csv('data.csv')
dataset = DataFrame(data,columns=['distance1','distance2','distance3','relativediff1','relativediff2','class','Absdiff1','Absdiff2','speed'])

array = dataset.values
X_test = array[:,3:5]
Y_test = array[:,5]
X_test2 = array[:,6:8]
Y_test2= array[:,8]

filename = 'finalized_model.sav'
# load the model from disk
global loaded_model
#filename = filename.astype('ITYPE_t')
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

filename = 'finalized_model2.sav'
# load the model from disk
global loaded_model2 
loaded_model2 = joblib.load(filename)
result2 = loaded_model2.score(X_test2, Y_test2)
print(result2)

#data collector
import time
import RPi.GPIO as GPIO
import requests
import pandas as pd
import numpy as np
request = None

#GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
PIN_TRIGGER = 17
PIN_ECHO = 13
duration = 0
GPIO.setup(PIN_TRIGGER,GPIO.OUT)
GPIO.setup(PIN_ECHO,GPIO.IN)
start_time = time.time()

number_instance = 9
global number_distance 
number_distance = 3
list_distance = np.zeros((number_distance), dtype = object)

classname = ['standstill','move further away','move nearer']
number_instanceperclass = number_instance/len(classname)
currentclass = classname[0]
global start 
start = time.time()
global h
h=1
distance = 0
while True:
#     print("collecting distance....")
    for j in range(number_distance):
        
        GPIO.output(PIN_TRIGGER,GPIO.LOW)
        time.sleep(0.2)
        
        GPIO.output(PIN_TRIGGER,GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(PIN_TRIGGER,GPIO.LOW)
        
        while GPIO.input(PIN_ECHO)==0:
                pulse_start_time = time.time()
                
        while GPIO.input(PIN_ECHO)==1:
                pulse_end_time = time.time()
                
           
        pulse_duration = pulse_end_time-pulse_start_time
        distance = round(17500*pulse_duration,2)
        print(distance)
        #check for distance and record video
        
            
        list_distance[j] = distance 
    # make predictions
#     print("finished")
    if distance <80:
            print("person detected")
            print("start recording video for 10 seconds")
            P1 = Process(target=video_recorder, args=(10,))
            P1.start()
            print("start posting information to telegram")
            P2 = Process(target=send_telegram, args=(list_distance,))
            P2.start()    
            P1.join()
            P2.join()
    
# newrows = pd.DataFrame(list_distance, columns =['distance1', 'distance2', 'distance3','class'], dtype = object)
# newrows.to_csv(r'data.csv', index = False)