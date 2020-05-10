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
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.input_blob = None
        self.output_blob = None
        self.network_plugin = None
        self.inference_handler = None
        self.ie = None
        self.device = None
        

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None, ie=None ):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0]+'.bin'
        self.net = IENetwork(model = model_xml, weights = model_bin)
        self.ie = IECore()
        
        if "CPU" in device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.info("Unsupported Layers Found before Applying Extension!")
                log.info(unsupported_layers)
        
        if cpu_extension and 'CPU' in device:
            self.ie.add_extension(cpu_extension, 'CPU')
            
        if "CPU" in device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("Unsupported Layers Found Even After Applying Extension!")
                log.error(unsupported_layers)
                sys.exit(1)
    
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        
        if num_requests == 0:
            self.network_plugin = self.ie.load_network(self.net, device)
        else:
            self.network_plugin = self.ie.load_network(self.net, device, num_requests)
        
        return self.get_input_shape()

    def get_input_shape(self):
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        self.inference_handler = self.network_plugin.start_async(request_id=request_id, 
                                                                 inputs={self.input_blob:frame})
        ###  Return any necessary information ###
       
        return self.network_plugin

    def wait(self, request_id):
        ###  Wait for the request to be complete. ###
        wait_in_interface = self.network_plugin.requests[request_id].wait(-1)
        ###  Return any necessary information ###
        
        return wait_in_interface

    def get_output(self, request_id, prev_output=None):
        ###  Extract and return the output results
        if prev_output:
            res = self.inference_handler.outputs[prev_output]
        else:
            res = self.network_plugin.requests[request_id].outputs[self.output_blob]
        
        return res
