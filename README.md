# :lungs: Chest-X-Ray Classification

![Custom badge](https://img.shields.io/badge/code%20style-PEP%208-brightgreen)

This project classifies chest X-ray images using a customized DenseNet121 neural network model.

## Contents

- [Technologies](#technologies)
- [Dataset description](#dataset-description)
- [Data preprocessing](#data-preprocessing)
- [Proposed model](#proposed-model)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Credits](#credits)

## Technologies

![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/tensorflow-f5f6f7?style=for-the-badge&logo=tensorflow&logoColor=ff7900)
![Keras](https://img.shields.io/badge/keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Sklearn](https://img.shields.io/badge/scikit--learn-3498cb?style=for-the-badge&logo=scikit-learn&logoColor=f89a36)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-222222?style=for-the-badge&logo=jupyter&logoColor=b84600)


## Dataset description

This dataset of chest X-ray (CXR) images was sourced from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

It contains 3616 COVID, 6012 Lung Opacity, 10192 Normal, and 1345 Viral Pneumonia CXR images.

<img height="400" src="figures/bar_chart.png" width="400"/><img height="400" src="figures/pie_chart.png" width="400"/>

Every image was PNG, grayscale, and 299 x 299 pixels.

<img height="200" src="figures/random_images.png" width="800"/>

## Data preprocessing

Images were processed using `tf.keras.applications.resnet50.preprocess_input`, which converts images from RGB to BGR
and then zero-centers each color channel with respect to the ImageNet dataset without scaling. The images were also
resized to 224 x 224 for ImageNet compatibility reasons.

In addition, real-time data augmentation was done. The table below details the parameters used.

<img height="180.5" src="figures/aug_parameters_summary.png" width="509"/>

For visualization purposes, some augmented pictures were saved to aug_images folder via the `save_to_dir` arg.

<img height="160" src="aug_images\aug_1128_8814508.png" width="160"/><img height="160" src="aug_images\aug_6868_6483323.png" width="160"/><img height="160" src="aug_images\aug_7903_3196687.png" width="160"/><img height="160" src="aug_images\aug_9366_9241711.png" width="160"/><img height="160" src="aug_images\aug_4010_3591919.png" width="160"/>

## Proposed model

New bottleneck layers and a classifier head were added to a pretrained DenseNet121 model.

```python
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_tensor=Input(shape=(IMG_DIMS, IMG_DIMS, 3)),
    input_shape=(IMG_DIMS, IMG_DIMS, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(4, activation='softmax'))
```

The base model layers were frozen,
resulting in 52,141,828 trainable and 7,140,712 non-trainable parameters.

## Methodology

The table below summarizes the dataset partition details.

<img height="153" src="figures/dataset_partition.png" width="590"/>

Images and were trained for 25 epochs with a learning rate of 1e-4.
The learning rate was lowered to 1e-5 for the last 5 epochs.

## Results

### Training Results

<img height="300" src="figures/categorical crossentropy loss.png" width="400"/><img height="300" src="figures/categorical accuracy.png" width="400"/>
<img height="300" src="figures/precision.png" width="400"/><img height="300" src="figures/recall.png" width="400"/>

### Testing Results

Output predictions were generated for the test input samples
and  compared to the true label values.

See `sklearn.metrics.classification_report` and `sklearn.metrics.ConfusionMatrixDisplay` for more details.

<img height="222.5" src="figures/classification_report.png" width="471.5"/>
<img height="533.3" src="figures/confusion_matrix.png" width="800"/>

## Requirements

Third party imports: matplotlib, numpy, IPython, PIL, sklearn, tensorflow, splitfolders

## Credits

- https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- https://arxiv.org/pdf/1608.06993.pdf
