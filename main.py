'''
AUTHOR: SHAIK DAVOOD.

PROJECT 1: INTEL EDGE AI FOR IOT DEVELOPERS NANODEGREE
'''

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-fe", "--frames_ignore", type=int, default=9, 
                        help="Number of Frames to ignore before counting a person (default: 9)")
    parser.add_argument("-al", "--enable_alert_limit", type=int, default=None,
                        help="Enable Intruder Alert Limit when person in frames goes up (default: None)")
    
    return parser


def connect_mqtt():
    client = mqtt.Client()

    return client

def draw_frame_on_inference(frame, result):
    '''
    Drawing the Bounding Boxes over the output frames.
    Counting the Number of People from the Detections per frame. 
    params:
    frame: frame of a video or an image to draw the bounding boxes.
    result: result from the DetectionOutputLayer of an SSD Network.
    '''
    current_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * init_w)
            ymin = int(obj[4] * init_h)
            xmax = int(obj[5] * init_w)
            ymax = int(obj[6] * init_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,155, 55), 3)
            current_count = current_count + 1
    return frame, current_count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    global init_w, init_h, prob_threshold
    current_req_num = 0
    total_count = 0
    latest_count = 0
    previous_count = 0
    duration_sum = 0
    duration_in_frame=0.0
    frame_count = 0
    infer_frame_count = 0
    single_image_mode = False
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    client.connect(HOSTNAME, port=MQTT_PORT, keepalive=60, bind_address=IPADDRESS)
    ### Load the model through `infer_network` ###
    n,c,h,w = infer_network.load_model(args.model, args.device, 1, 1, current_req_num, args.cpu_extension)

    ### Handle the input stream ###
    if args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input  
        
    capture_frames = cv2.VideoCapture(input_stream)
    length_of_video = int(capture_frames.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(capture_frames.get(cv2.CAP_PROP_FPS))
        ### Read from the video capture ###
    infer_time_start = time.time()
    if input_stream:
        capture_frames.open(args.input)
    if not capture_frames.isOpened():
        log.error("Unable to Open the Video File.")
    
    init_w = capture_frames.get(3)
    init_h = capture_frames.get(4)
    out =  cv2.VideoWriter(os.path.join("people_counter.mp4"), 0x00000021, frame_rate, (int(init_w), int(init_h)), True)
    while capture_frames.isOpened():

        isEnd, frame = capture_frames.read()
        frame_count = frame_count + 1
        current_count = 0
        if not isEnd:
            break
        cv2.waitKey(10)
         ### Pre-process the image as needed ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2,0,1))
        image = image.reshape((n,c,h,w))

        # Starting the Asynchronous Inference: 
        inf_start = time.time()
        infer_network.exec_net(current_req_num, image)

        ### Waiting for the result ###
        if infer_network.wait(current_req_num) == 0:
            duration = (time.time() - inf_start)
            results = infer_network.get_output(current_req_num)
            out_frame, current_count = draw_frame_on_inference(frame, results) 
            duration_message = "Inference Time Per Frame: {:.3f}ms".format(duration * 1000)
            
        if current_count > 0:
            infer_frame_count = infer_frame_count + 1
            duration_sum = duration_sum + float(infer_frame_count)/frame_rate
        
        if current_count > 0 and infer_frame_count > args.frames_ignore and previous_count > 0:
            '''
            If the Count of People Goes up and keeps like that for more than the expected output 
            '''
            previous_count = max(previous_count, current_count)
            
                
        if previous_count == 0 and infer_frame_count > args.frames_ignore:
            total_count = total_count + current_count
#             infer_frame_count = 0
            previous_count = max(previous_count, current_count)
            client.publish("person", json.dumps({"count": current_count}))
            client.publish("person", json.dumps({"total": total_count}))
            
            
        
        if previous_count != 0 and current_count == 0:
            duration_in_frame = infer_frame_count/frame_rate
            for i in range(previous_count):
                client.publish("person/duration", json.dumps({"duration": duration_in_frame}))
             
        if current_count == 0:
            infer_frame_count = 0
            previous_count = current_count
            duration_sum = 0.0  
            client.publish("person", json.dumps({"count": current_count}))
        cv2.putText(out_frame, duration_message, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (210, 10, 10), 1)
        
    

        
        out.write(out_frame)
        
        client.publish("person", json.dumps({"count": current_count}))
       
        
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()  

        ### Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imWrite('infer_out.jpg', frame)
                       
    capture_frames.release()
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
