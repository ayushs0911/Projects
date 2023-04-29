## Emotion Detection 

**Problem Statement** : <br>
- Emotion detection is a problem in deep learning where the goal is to identify the emotion expressed in a given text, image, or speech signal.
- The task is to classify the emotion into one of several categories, such as happiness, sadness, anger, fear, surprise, or neutral. Emotion detection has applications in a variety of fields, including psychology, market research, social media analysis, and customer service. 
- The challenge lies in accurately capturing the nuances and complexities of human emotions, which can vary widely across individuals, cultures, and contexts. 
- Deep learning approaches, such as CNNs, RNNs, and transformer-based models, have shown promising results in solving this problem by leveraging large amounts of labeled data and powerful computational resources.<br>

**Data Source : Kaggle**<br>
Directories :

```
There are 0 directories and 7164 images in '/content/dataset/images/train/happy'.
There are 0 directories and 3993 images in '/content/dataset/images/train/angry'.
There are 0 directories and 3205 images in '/content/dataset/images/train/surprise'.
There are 0 directories and 4982 images in '/content/dataset/images/train/neutral'.
There are 0 directories and 436 images in '/content/dataset/images/train/disgust'.
There are 0 directories and 4103 images in '/content/dataset/images/train/fear'.
There are 0 directories and 4938 images in '/content/dataset/images/train/sad'.
```

<!-- **Data Visualization** -->

 **Models**
 - `Lenet` 
 - `ResNet34`
 - Transfer Learning `EfficientNet`
 - FineTuning `EfficientNet`
 - `Vision Transformer`
 - Using `HuggingFace Transformer`
 
 **Best Performing Model Architecture** : HuggingFace downloaded Model performed best. 
 **Installations**
 ```
 !pip install transformers
 ```
 **Model Download and Finetuning on our dataset**
 ```
 
from transformers import ViTFeatureExtractor, TFViTModel


base_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

inputs = Input(shape = (256,256,3))
x = resize_rescale_hf(inputs)
x = base_model.vit(x)[0][:,0,:]
#print(x)
output = Dense(7, activation = 'softmax')(x)

hf_model = tf.keras.Model(inputs=inputs, outputs=output)
```
**Training**
```
history_hf = hf_model.fit(train_dataset,
                               epochs =20,
                               validation_data = val_dataset,
                               callbacks = [es_callback,
                                            plateau_callback,
                                            model_checkpoint,
                                            create_tensorboard_callback('trainning_logs', 'hf_model')])
```
```
Saving TensorBoard log files to: trainning_logs/hf_model/20230429-030524
Epoch 1/20
901/901 [==============================] - 1271s 1s/step - loss: 0.7406 - accuracy: 0.7293 - val_loss: 0.8703 - val_accuracy: 0.6903 - lr: 5.0000e-05
Epoch 2/20
901/901 [==============================] - 1265s 1s/step - loss: 0.5019 - accuracy: 0.8237 - val_loss: 0.9272 - val_accuracy: 0.6874 - lr: 5.0000e-05
Epoch 3/20
901/901 [==============================] - 1267s 1s/step - loss: 0.2883 - accuracy: 0.9065 - val_loss: 1.0950 - val_accuracy: 0.6797 - lr: 5.0000e-05
Epoch 4/20
901/901 [==============================] - 1268s 1s/step - loss: 0.1896 - accuracy: 0.9369 - val_loss: 1.2746 - val_accuracy: 0.6817 - lr: 5.0000e-05
Epoch 4: early stopping
```
 
 **Saving Model to Google Drive**
 
```
from google.colab import drive
drive.mount('/content/gdrive')

# Save the trained model to a file
save_model(hf_model, os.path.join('/content/gdrive/My Drive', 'hf_model.h5'))
```
 
