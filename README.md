# what_is_in_my_plate
---
This project presents a web application that utilizes `transfer learning` with the `EfficientNetB0` architecture to classify food items in images. The main objective was to surpass the performance achieved in the DeepFood paper. The model was trained on the Food101 dataset, which comprises 101,000 images spanning 101 distinct food categories.

<p align="center">
    <img src="https://github.com/rajtiwariee/helper_functions/blob/master/assets/model%20prediction.png?raw=true">
</p>

![image](https://github.com/rajtiwariee/what_is_in_my_plate/assets/98082499/d7c4727d-39c2-46c0-a34d-cfb3924cfbe8)

## Data Acquisition:
---
The Food101 dataset was employed for training the model. TensorFlow Datasets library facilitated the process of loading and utilizing the dataset effectively.

## Model Architecture:
---
Transfer learning was employed using the EfficientNetB0 architecture. This pre-trained model had initially been trained on the ImageNet dataset. By fine-tuning the model on the Food101 dataset, the accuracy was significantly improved.


## Future Enhancements:
---
Furthermore, increasing the model's accuracy is a priority, and this will involve training it on a larger and more diverse dataset.

## Tools :
The following tools were utilized in the development of this project:

* Python
* Jupyter Notebook
* TensorFlow
* Keras

I would like to thank Daniel Bourke for his helpful tutorials and resources.
