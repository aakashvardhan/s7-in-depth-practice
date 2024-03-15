# Efficient CNN for digit classification on MNIST dataset

- Step 1: (Iteration 1) [#iteration-1]
- Step 2: (Iteration 2-5) [#iteration-2]
- Step 3: (Iteration 6-8) [#iteration-3]

## Introduction

Our aim is to design an efficient Convolutional Neural Network (CNN) for digit classification on the MNIST dataset. The target is to achieve a test accuracy of 99.4% with a model that has less than 8,000 parameters and less than 15 epochs.

## Code Iterations

We have made several iterations to reach our target. The iterations are as follows:

Link to model and its utilities: [Model Folder](https://github.com/aakashvardhan/s7-in-depth-practice/tree/main/models)

### Iteration 1

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/model1_train.ipynb)

#### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
              ReLU-3           [-1, 10, 26, 26]               0
            Conv2d-4           [-1, 10, 24, 24]             900
       BatchNorm2d-5           [-1, 10, 24, 24]              20
              ReLU-6           [-1, 10, 24, 24]               0
            Conv2d-7           [-1, 20, 22, 22]           1,800
       BatchNorm2d-8           [-1, 20, 22, 22]              40
              ReLU-9           [-1, 20, 22, 22]               0
        MaxPool2d-10           [-1, 20, 11, 11]               0
           Conv2d-11           [-1, 10, 11, 11]             200
      BatchNorm2d-12           [-1, 10, 11, 11]              20
             ReLU-13           [-1, 10, 11, 11]               0
           Conv2d-14             [-1, 10, 9, 9]             900
      BatchNorm2d-15             [-1, 10, 9, 9]              20
             ReLU-16             [-1, 10, 9, 9]               0
           Conv2d-17             [-1, 20, 7, 7]           1,800
      BatchNorm2d-18             [-1, 20, 7, 7]              40
             ReLU-19             [-1, 20, 7, 7]               0
           Conv2d-20             [-1, 10, 7, 7]             200
      BatchNorm2d-21             [-1, 10, 7, 7]              20
             ReLU-22             [-1, 10, 7, 7]               0
           Conv2d-23             [-1, 10, 1, 1]           4,900
================================================================
Total params: 10,970
Trainable params: 10,970
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.61
Params size (MB): 0.04
Estimated Total Size (MB): 0.65
----------------------------------------------------------------
```

- **Target**: Design a basic skeleton of the model and train it to check the training pipeline. This involves setting up the data pipeline, defining the model, loss function, optimizer, and training loop.

- **Results**: 
  - Parameters: 10,970
  - Best Train Accuracy: 99.73%
  - Best Test Accuracy: 99.10%

- **Analysis**:
    - The model is overfitting as the training accuracy is higher than the test accuracy.
    - The model has more than 8,000 parameters.

### Iteration 2

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/model2_train.ipynb)

- **Changes**:
  - Created a modular structure for the model.
    - Created a `ConvBlock` class to define a sequence of convolution layers followed by ReLU Activation, Batch Normalization, and Dropout.
    - Dropout is optional and can be enabled by setting the `dropout_value` parameter of `ConvBlock` to a non-zero value.
    - Created a `TransitionBlock` class to allow for downsampling the feature map size using a 1x1 convolution layer followed by a 2x2 max pooling layer.
  - Added Global Average Pooling (GAP) layer at the end of the model to reduce the feature map size to 1x1.

#### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
         ConvBlock-5            [-1, 8, 26, 26]               0
            Conv2d-6           [-1, 16, 24, 24]           1,152
              ReLU-7           [-1, 16, 24, 24]               0
       BatchNorm2d-8           [-1, 16, 24, 24]              32
           Dropout-9           [-1, 16, 24, 24]               0
        ConvBlock-10           [-1, 16, 24, 24]               0
           Conv2d-11           [-1, 16, 22, 22]           2,304
             ReLU-12           [-1, 16, 22, 22]               0
      BatchNorm2d-13           [-1, 16, 22, 22]              32
          Dropout-14           [-1, 16, 22, 22]               0
        ConvBlock-15           [-1, 16, 22, 22]               0
           Conv2d-16            [-1, 8, 22, 22]             128
        MaxPool2d-17            [-1, 8, 11, 11]               0
  TransitionBlock-18            [-1, 8, 11, 11]               0
           Conv2d-19              [-1, 8, 9, 9]             576
             ReLU-20              [-1, 8, 9, 9]               0
      BatchNorm2d-21              [-1, 8, 9, 9]              16
          Dropout-22              [-1, 8, 9, 9]               0
        ConvBlock-23              [-1, 8, 9, 9]               0
           Conv2d-24             [-1, 16, 7, 7]           1,152
             ReLU-25             [-1, 16, 7, 7]               0
      BatchNorm2d-26             [-1, 16, 7, 7]              32
          Dropout-27             [-1, 16, 7, 7]               0
        ConvBlock-28             [-1, 16, 7, 7]               0
           Conv2d-29             [-1, 16, 7, 7]           2,304
             ReLU-30             [-1, 16, 7, 7]               0
      BatchNorm2d-31             [-1, 16, 7, 7]              32
          Dropout-32             [-1, 16, 7, 7]               0
        ConvBlock-33             [-1, 16, 7, 7]               0
        AvgPool2d-34             [-1, 16, 1, 1]               0
           Conv2d-35             [-1, 10, 1, 1]             160
================================================================
Total params: 8,008
Trainable params: 8,008
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.98
Params size (MB): 0.03
Estimated Total Size (MB): 1.02
----------------------------------------------------------------
```

- **Results**: 
  - Parameters: 8,008
  - Best Train Accuracy: 99.14%
  - Best Test Accuracy: 99.05%

- **Analysis**:
    - The model is slightly overfitting, which indicates a decrease in generalization loss.
    - The model is close to 8,000 parameters but the test accuracy is still below 99.4%.

### Iteration 3

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/train_with_transformed.ipynb)

- **Changes**:
  - Added data augmentation to the training dataset using the `transforms` module from `torchvision`.
    - `RandomResizedCrop`: Randomly crop the image and resize it to (28, 28) at the scale of (0.8, 1.0).
    - `RandomRotation`: Randomly rotate the image by a maximum of 7 degrees.

- **Results**: 
  - Parameters: 8,008
  - Best Train Accuracy: 98.54%
  - Best Test Accuracy: 99.27%

- **Analysis**:
    - The model is not overfitting anymore.
    - The test accuracy is close to 99.4% but the model is still not able to achieve the target.

### Iteration 4

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/train_withLRScheduler.ipynb)

- **Changes**:
  - Added a learning rate scheduler to the training loop.
    - Used the `StepLR` scheduler from the `torch.optim.lr_scheduler` module.
    - Reduced the learning rate by a factor of 0.1 after every 7 epochs.

- **Results**:
    - Parameters: 8,008
    - Best Train Accuracy: 98.81%
    - Best Test Accuracy: 99.30%

- **Analysis**:
    - The model has slightly improved with the increase in test accuracy by 0.03%.

### Iteration 5

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/model3_with_img_aug.ipynb)

- **Changes**:
  - Reduced the number of parameters to 5,932 in the model by rearranging the layers and reducing the number of channels in the convolution layers.
  - Removed the `RandomResizedCrop` and `RandomRotation` data augmentation techniques from the training dataset.
  - Added new data augmentation techniques to the training dataset using the `transforms` module from `torchvision`.
    - `RandomAffine`: Randomly apply affine transformations to the image, translating the image by a maximum of 0.1, shearing the image by a maximum of 10 degrees, and scaling the image by a factor of (0.8, 1.2).
    - `ColorJitter`: Randomly change the brightness, contrast, saturation, and hue of the image.

#### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 26, 26]              36
              ReLU-2            [-1, 4, 26, 26]               0
       BatchNorm2d-3            [-1, 4, 26, 26]               8
           Dropout-4            [-1, 4, 26, 26]               0
         ConvBlock-5            [-1, 4, 26, 26]               0
            Conv2d-6            [-1, 8, 24, 24]             288
              ReLU-7            [-1, 8, 24, 24]               0
       BatchNorm2d-8            [-1, 8, 24, 24]              16
           Dropout-9            [-1, 8, 24, 24]               0
        ConvBlock-10            [-1, 8, 24, 24]               0
           Conv2d-11           [-1, 16, 22, 22]           1,152
             ReLU-12           [-1, 16, 22, 22]               0
      BatchNorm2d-13           [-1, 16, 22, 22]              32
          Dropout-14           [-1, 16, 22, 22]               0
        ConvBlock-15           [-1, 16, 22, 22]               0
           Conv2d-16            [-1, 8, 22, 22]             128
        MaxPool2d-17            [-1, 8, 11, 11]               0
  TransitionBlock-18            [-1, 8, 11, 11]               0
           Conv2d-19              [-1, 8, 9, 9]             576
             ReLU-20              [-1, 8, 9, 9]               0
      BatchNorm2d-21              [-1, 8, 9, 9]              16
          Dropout-22              [-1, 8, 9, 9]               0
        ConvBlock-23              [-1, 8, 9, 9]               0
           Conv2d-24             [-1, 16, 7, 7]           1,152
             ReLU-25             [-1, 16, 7, 7]               0
      BatchNorm2d-26             [-1, 16, 7, 7]              32
          Dropout-27             [-1, 16, 7, 7]               0
        ConvBlock-28             [-1, 16, 7, 7]               0
           Conv2d-29             [-1, 16, 7, 7]           2,304
             ReLU-30             [-1, 16, 7, 7]               0
      BatchNorm2d-31             [-1, 16, 7, 7]              32
          Dropout-32             [-1, 16, 7, 7]               0
        ConvBlock-33             [-1, 16, 7, 7]               0
        AvgPool2d-34             [-1, 16, 1, 1]               0
           Conv2d-35             [-1, 10, 1, 1]             160
================================================================
Total params: 5,932
Trainable params: 5,932
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.02
Estimated Total Size (MB): 0.73
----------------------------------------------------------------
```

- **Results**:
  - Parameters: 5,932
  - Best Train Accuracy: 98.38%
  - Best Test Accuracy: 99.38%

- **Analysis**:
    - The model has improved significantly with the test accuracy reaching 99.38%.
    - However, the model's accuracy was plateauing after 10 epochs, which indicates that the model has learned the dataset and is not improving further.

### Iteration 6

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/model3_with_reduceLRPlateau.ipynb)

- **Changes**:
  - Added a learning rate scheduler to the training loop.
    - Used the `ReduceLROnPlateau` scheduler from the `torch.optim.lr_scheduler` module.
    - Reduced the learning rate by a factor of 0.1 when the test loss did not improve for 2 epochs.
  - Followed by `Conv(1x1)`, `AvgPool2d` is used to reduce the number of channels to 10.

- **Results**:
  - Parameters: 5,932
  - Best Train Accuracy: 98.68%
  - Best Test Accuracy: 99.40%

- **Analysis**:
    - The model has achieved the target test accuracy of 99.4%
    - However, the test accuracy is not consistent and fluctuates around 99.4% or below that.

## Final Iteration

Link: [Notebook](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/notebooks/model3_with_increasedParam.ipynb)

- **Changes**:
  - Increased the number of parameters to 7,676 in the model.
  - Removed the `ReduceLROnPlateau` scheduler and used the `StepLR` scheduler from the `torch.optim.lr_scheduler` module.
  - Increased the learning rate to 0.04 and reduced the learning rate by a factor of 0.1 after every 7 epochs.

#### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 26, 26]              36
              ReLU-2            [-1, 4, 26, 26]               0
       BatchNorm2d-3            [-1, 4, 26, 26]               8
           Dropout-4            [-1, 4, 26, 26]               0
         ConvBlock-5            [-1, 4, 26, 26]               0
            Conv2d-6            [-1, 8, 24, 24]             288
              ReLU-7            [-1, 8, 24, 24]               0
       BatchNorm2d-8            [-1, 8, 24, 24]              16
           Dropout-9            [-1, 8, 24, 24]               0
        ConvBlock-10            [-1, 8, 24, 24]               0
           Conv2d-11           [-1, 16, 22, 22]           1,152
             ReLU-12           [-1, 16, 22, 22]               0
      BatchNorm2d-13           [-1, 16, 22, 22]              32
          Dropout-14           [-1, 16, 22, 22]               0
        ConvBlock-15           [-1, 16, 22, 22]               0
           Conv2d-16            [-1, 8, 22, 22]             128
        MaxPool2d-17            [-1, 8, 11, 11]               0
  TransitionBlock-18            [-1, 8, 11, 11]               0
           Conv2d-19             [-1, 16, 9, 9]           1,152
             ReLU-20             [-1, 16, 9, 9]               0
      BatchNorm2d-21             [-1, 16, 9, 9]              32
          Dropout-22             [-1, 16, 9, 9]               0
        ConvBlock-23             [-1, 16, 9, 9]               0
           Conv2d-24             [-1, 16, 7, 7]           2,304
             ReLU-25             [-1, 16, 7, 7]               0
      BatchNorm2d-26             [-1, 16, 7, 7]              32
          Dropout-27             [-1, 16, 7, 7]               0
        ConvBlock-28             [-1, 16, 7, 7]               0
           Conv2d-29             [-1, 16, 7, 7]           2,304
             ReLU-30             [-1, 16, 7, 7]               0
      BatchNorm2d-31             [-1, 16, 7, 7]              32
          Dropout-32             [-1, 16, 7, 7]               0
        ConvBlock-33             [-1, 16, 7, 7]               0
           Conv2d-34             [-1, 10, 7, 7]             160
        AvgPool2d-35             [-1, 10, 1, 1]               0
================================================================
Total params: 7,676
Trainable params: 7,676
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.73
Params size (MB): 0.03
Estimated Total Size (MB): 0.76
----------------------------------------------------------------
```

#### Results

```
EPOCH: 1
Loss=0.08434382826089859 Batch_id=468 Accuracy=86.18: 100%|██████████| 469/469 [00:46<00:00, 10.14it/s]
Test set: Average loss: 0.0588, Accuracy: 9827/10000 (98.27%)

EPOCH: 2
Loss=0.0978543683886528 Batch_id=468 Accuracy=96.30: 100%|██████████| 469/469 [00:42<00:00, 10.97it/s]
Test set: Average loss: 0.0355, Accuracy: 9887/10000 (98.87%)

EPOCH: 3
Loss=0.03732876852154732 Batch_id=468 Accuracy=96.96: 100%|██████████| 469/469 [00:42<00:00, 10.91it/s]
Test set: Average loss: 0.0367, Accuracy: 9872/10000 (98.72%)

EPOCH: 4
Loss=0.06504508852958679 Batch_id=468 Accuracy=97.27: 100%|██████████| 469/469 [00:43<00:00, 10.79it/s]
Test set: Average loss: 0.0288, Accuracy: 9909/10000 (99.09%)

EPOCH: 5
Loss=0.12743593752384186 Batch_id=468 Accuracy=97.59: 100%|██████████| 469/469 [00:42<00:00, 11.01it/s]
Test set: Average loss: 0.0338, Accuracy: 9886/10000 (98.86%)

...
...
...

EPOCH: 10
Loss=0.031964439898729324 Batch_id=468 Accuracy=98.37: 100%|██████████| 469/469 [00:43<00:00, 10.84it/s]
Test set: Average loss: 0.0182, Accuracy: 9945/10000 (99.45%)

EPOCH: 11
Loss=0.13113215565681458 Batch_id=468 Accuracy=98.40: 100%|██████████| 469/469 [00:42<00:00, 10.97it/s]
Test set: Average loss: 0.0168, Accuracy: 9943/10000 (99.43%)

EPOCH: 12
Loss=0.05126373469829559 Batch_id=468 Accuracy=98.34: 100%|██████████| 469/469 [00:43<00:00, 10.76it/s]
Test set: Average loss: 0.0171, Accuracy: 9943/10000 (99.43%)

EPOCH: 13
Loss=0.06044589355587959 Batch_id=468 Accuracy=98.41: 100%|██████████| 469/469 [00:43<00:00, 10.88it/s]
Test set: Average loss: 0.0174, Accuracy: 9948/10000 (99.48%)

EPOCH: 14
Loss=0.02426617406308651 Batch_id=468 Accuracy=98.37: 100%|██████████| 469/469 [00:43<00:00, 10.69it/s]
Test set: Average loss: 0.0171, Accuracy: 9942/10000 (99.42%)

EPOCH: 15
Loss=0.0379183366894722 Batch_id=468 Accuracy=98.49: 100%|██████████| 469/469 [00:43<00:00, 10.78it/s]
Test set: Average loss: 0.0174, Accuracy: 9944/10000 (99.44%)
```

![Accuracy and Loss](https://github.com/aakashvardhan/s7-in-depth-practice/blob/main/final-model-performance.png)

- **Results**:
  - Parameters: 7,676
  - Best Train Accuracy: 98.49%
  - Best Test Accuracy: 99.44%

- **Analysis**:
    - The model has achieved the target test accuracy of 99.4% consistently.
    - The model has learned the dataset and is not overfitting or underfitting.



