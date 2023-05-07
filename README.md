# Advanced Business Analytics Assignment 2
## Food Classification

Given a unlabeled dataset of restaurant images 
(food and interior) the task is to train a model
to classify between food and interior. 
First, the given unlabeled dataset is first
manually assiged labels (food or interior).
Second, the following models are fine‐tuned on the 
labeled dataset: Alexnet, VGG, ResNet,
Vision Transformer (ViT) and ConvNeXT.
ViT is chosen as a final model based on the
performance of the models in terms of the validation 
\accuracy. Moreover, further experiments are conducted
to improve the performance of ViT. Finally, classification metrices are computed for the chosen model as a final assessment. The final chosen model achieves an accuracy of 92.91% on validation and 92.46% on test sets.

## Dataset 

The dataset consists of 117K unlabeled images of restaurants. The images are manually labels as food or interior. The whole dataset is randomly divided into training (80\%), validation (10\%) and test (10\%) sets. The dataset is unbalanced. \(70.76\%\) of the training set belongs to class "food", while \(29.23\%\) belongs to class "interior".

## Methods

Twelve experiments are conducted in 
total using different architectures such 
as Alexnet, VGG, vision transformers as backbones. 
Vision transformers trained with 
Weighted Cross-Entropy and Mini Batch Gradient 
Descent with learning rate of 0.001 a
chieved the best performance on the validation set
(92.91% accuracy). 
Its performance was further assessed on the test 
set using precision, recall and AUC. 
The model shows a descent performance on the test set:
Accuracy=0.9246, Precision=0.9472, 
Recall=0.9457, AUC=0.9088.