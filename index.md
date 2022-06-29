**Introduction To Intel Distribution Of  OpenVINO™ Toolkit for Deep Learning**

**Accelerate Deep Learning Inference with  OpenVINO™ Toolkit**

![33](https://user-images.githubusercontent.com/75186414/176487352-0ade5f33-789b-4a47-af7a-0426f88c2b84.jpeg)


**
Introduction to  OpenVINO™**

Open Visual Inference and Neural network Optimization(OpenVINO) is an open-source toolkit developed by Intel. It is a set of libraries, optimization tools, and information resources that can be used to fast-track the development of cost-effective computer vision and deep learning applications which emulate human vision.

Intel has developed an open API (OpenVINO) that allows us to execute a model across the devices without understanding the hardware details or the underlying libraries required to access them. Software developers and Data Scientists who work on computer vision can use this product to help speed up their solutions on a variety of platforms, including CPU, GPU, VPU, and FPGA. In short, One can optimize and deploy their trained neural networks and Deep Learning models quickly at the Edge.

**What’s inside the OpenVINO™ toolkit?**

**Model Optimizer:** It is a command-line tool that uses static model analysis for optimal execution of the deep learning model by adjusting the network. It produces an Intermediate Representation (IR) of the model which can be used as the input for the Inference engine.

We’ll look at three optimization techniques that OpenVINO provides to make the model faster. They include Quantization, Freezing, and Fusion.

**Quantization**

In OpenVINO, the weights and biases of the pre-trained models are available in a variety of precisions such as FP32, FP16, and INT8.

The higher precision model may produce better accuracy, but it is more complex and takes up more space, demanding a lot more computing power. Lowering model precision reduces model accuracy because we lose some critical information when we reduce model precision, resulting in a reduction in accuracy.

**Freezing**

Specific layers in training are frozen so that you can fine-tune and train on only a sample of them. It’s used as part of a larger model, notably Tensorflow models. When TensorFlow models are frozen, certain actions and metadata that are only needed for training are removed. Backpropagation, for example, is only necessary during training and not inference.

**Fusion**

In a nutshell, fusion means combing multiple-layer operations into a single operation. For TensorFlow topologies, grouped convolution fusing is a specialized optimization technique. The primary idea behind this optimization is to aggregate the convolutional results for the Split outputs and then recombine them in the same order as they came out of Split using a concatenation operation.

**Intermediate Representation (IR):** It has two files to describe the model (the ‘.XML’ contains key metadata and the ‘.bin’ file consists of the model’s weights and biases in binary format).

**Inference Engine:** The Inference Engine (IE) is a collection of C++ packages that provide a common API that allows users to do inference on either CPU or any other device. It provides an API for reading the MO’s IR files (.bin and.xml), setting inputs and outputs, and running the model on devices. It reads the Intermediate Representation and runs the model on the device of your choice. Python bindings are also offered in addition to the primary C++ implementation.

**Model Zoo:** It consists of a wide range of free, pre-trained deep learning models developed by Intel. Have a look at both Intel and the public models and choose the right model for your solution. The purpose of Model Zoo is to execute, train, and deploy Intel-optimized models efficiently. These models can be used for Image Recognition, Language Modeling, Image Segmentation, Object Detection, etc.

**What is the workflow for deploying the model using OpenVINO™?**

Due to the multiple frameworks used for numerous reasons, the deployment procedure may be difficult. Another issue that complicates the process is that inference might be done on platforms that are limited in terms of hardware and software; thus, the training framework should not be used. Alternatively, specific hardware-optimized inference APIs could be used.

OpenVINO toolkit includes two primary components: Model Optimizer and Inference Engine, which allow developers to maximize the performance of their pre-trained models.

![image](https://user-images.githubusercontent.com/75186414/176489432-a33b7760-cfd8-4514-97f0-1d3dbc6a9071.png)

**What are the benefits of using OpenVINO™?**

In layman’s terms, you should use OpenVINO™ when you’d like to run many models on different devices at the same time

**Installation**

Look over the system requirements and install the toolkit to set up the Environment

To develop the model and obtain the optimization toolkit, install OpenVINO™ Development Tools by using the following command.

```
!pip install openvino-dev[tensorflow2]==2021.4.2
```
To start with Tensorflow and Keras, for an instance, We will train a Convolutional Neural Network model on the MNIST dataset, to categorize the handwritten digits into one of ten classes. We use Keras API to load the MNIST dataset.

_Import the libraries and load the dataset._

```
from openvino.inference_engine import IECore
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
data=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=data.load_data()
```

_Develop a sequential model by adding convolutional and maxpool layers to it and fit the model after compiling._

```
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2)))
model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D( pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))
```
After compiling and fitting the model, convert the model to OpenVINO IR Format by calling the model optimizer.

_Set the paths of the source and converted models._

```
model_name = "model"
model_path = Path(model_name)
ir_path = Path(model_path).with_suffix(".xml")
ir_name = "model_ir"
```

_Execute the command for model optimizer_

We can use the ```
--reverse-input-channels```option to make the converted model work with BGR images. To set the static shape, We can use the ```
--input_shape parameter``` 

```
mo_command = f"""mo
                 --saved_model_dir {model_fname}        
                 --input_shape "[1,28,28,1]"
                 --data_type FP16
                 --output_dir "{model_path.parent}"
                 --model_name "{ir_name}"
                 """              
                 
mo_command = " ".join(mo_command.split())
print("Converting the model from TensorFlow to OpenVINO:")
display(Markdown(f"`{mo_command}`"))```
```


_Run the Model Optimizer by the following command to convert the model to IR format. Model Optimizer creates three files namely model_ir.bin, model_ir.xml and model_ir.mapping._

```
mo_result = %sx $mo_command
print("\n".join(mo_result))
```

_Following the Model Optimization, the output will be displayed as follows._

![image](https://user-images.githubusercontent.com/75186414/176494919-ca212cc0-abe7-4c94-83f5-0b3f484d07b0.png)

_Load the converted model and initialize the inference engine by calling IECore(). Then compile the model._

```
ie = IECore()
final_model = ie.read_network(model=model_xml, weights=model_bin)
compiled_network=ie.load_network(network=final_model,
device_name="CPU")
```
_At the end, we can test the model on test.csv and get the submission file._

```
for test_img in test_imgs:
    input_layer = next(iter(compiled_network.input_info))
    output_layer = next(iter(compiled_network.outputs))
    res = compiled_network.infer(inputs={input_layer: test_img})
    res = res[output_layer]
```


**Conclusion** 

Deep learning-based models are simple to train on NVIDIA GPUs, thanks to a large number of popular frameworks that support them. However, executing them most efficiently on devices with limited resources is not easy. Because Intel owns and manufactures the vast majority of computing devices, there was a clear need to create something that covers all of their devices for inference. As a result, they built OpenVINO™, which provides a uniform API for accessing all of them at a high level.

In this article, We have seen a simple example where we have used the OpenVINO™ toolkit to run a Tensorflow model.

**#oneAPI**

