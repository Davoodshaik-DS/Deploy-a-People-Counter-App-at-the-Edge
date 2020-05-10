# Udacity | Intel (R) Edge AI for IoT Developers Nanodegree
### PROJECT 1: DEPLOY A PEOPLE COUNTER APP AT THE EDGE
##### SHAIK DAVOOD

## Explaining Custom Layers

Custom layers are neural network model layers that are not natively supported by a given model framework.They depend on the framework we use

The process behind converting custom layers involves two necessary custom layer extensions Custom Layer Extractor (responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial) and Custom Layer Operation (responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters)

Some of the potential reasons for handling custom layers are

When a layer isn’t supported by the Model Optimizer ,Model Optimizer does not know about the custom layers so it needs to taken care of and also need to handle for handle unsupported layers at the time of inference.
allow model optimizer to convert specific model to Intermediate Representation.
## Comparing Model Performance

For model comparison, I choose different approachs for each of the following metrics, as they need to be evaluated on their own
Accuracy:
As far as with my app is related, both TensorFlow model and converted OpenVINO one perform exactly the same. This is mostly because the logic I used inside the app tries to be "tolerant" with misdetections.
This means OpenVINO model is smaller while compared to other models, a great improvement considering usual memory restrictions in IoT devices.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
*hospitals:we can us ethis kind of app in hospital to moniter and give some announcement(as a alert) to avoid crowd which leads to infect the disease to each other.
*in railways station:to give alert on approching the people towards the track
*in metro train station:to count and allow the number of people can enter the metro train to avoid suffocation

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

*Better be the model accuracy more are the chances to obtain the desired results through an app deployed at edge.
*Lighting here refers to lighter the model more faster it will get execute and more adequate results in faster time as compared to a heavier model.
*Focal length/image also have a effect as better be the pixel quality of image or better the camera focal length,more clear results ww will obtain.
## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD Inception V2]
  -model source 
*[http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  -extraction
  *tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
  *rm -r ssd_inception_v2_coco_2018_01_28.tar.gz
- I converted the model to an Intermediate Representation with the following arguments...
  * /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channel -o ssd_inception
  
-time taken for generating IR model: 54.19 seconds

-To run the model 
*python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssd_inception/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

- The model was insufficient for the app because It has issues like first model where it failed to detect person in specific specific seconds .when second person came it didn't worked for a few seconds. Similarly for 3rd person it didn't worked again it failed.thus average duration may not be optained exactly so we cann't use this model as well.
  
-- I tried to improve the model for the app by checking in documentation and I found that shape attribute is required.So I did that again with shape attribute as well
  
  


- Model 2: [ssd_mobilenet_v1_coco]
  - [Model Source]
  * wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  -extracting the file
  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  rm -r ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments...
  * /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v1_coco_2018_01_28/pipeline.config --reverse_input_channel -o ssd_mbl_v1
  -time taken for generating IR model: 51.81 seconds
  -To run the model 
  * python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssd_mbl_v1/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  - The model was insufficient for the app because 
  - I tried to improve the model for the app by checking in documentation and I found that shape attribute is required.So I did that again with shape attribute as well

- Model 3: [ssdlite_mobilenet_v2_coco]
  - [Model Source]
  * wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  *tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  * rm -r ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
  
  - I converted the model to an Intermediate Representation with the following arguments...
  /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --reverse_input_channel -o ssdlite_mobilenet
  -time taken for generating IR model: 53.68 seconds
  -To run the model 
  * python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/ssdlite_mobilenet/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  - The model was insufficient for the app because The model not was quite appropriate in terms of detecting the people facing backwards althought the inference speed was very fast inference latency was around 25 milliseconds.But still in some intermediate frames in some videos sometimes it fails to properly draw the bounding boxes. In the app to accomodate this issue I have implemented to igonre such intermediate miscalculations
 
 
 
-i have also gave a try to most of the models in tensorflow model zoo https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md 
where i noticed error and I tried to improve the model for the app by checking in documentation and I found that shape attribute is required.So I did that again with shape attribute as well.Still it give errors.
-one more thing is while comparing RCNN and SSD models RCNN models are slower than SSd models


#conclusion:
-I founded my suitable model from this site: https://docs.openvinotoolkit.org/2019_R3/_models_intel_index.html

The model I found suitable was:(intel person detection retail 0013)

https://docs.openvinotoolkit.org/2019_R3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html

-the commands I used to download the model are:

*cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
*sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace

after downloading check the .bin and .xml files in following directory:
*/home/workspace/intel/person-detection-retail-0013/FP16

The command I used to run main.py is:

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm



this above intel pretrained model detected the exact count of people in the video and the number of people on the screen