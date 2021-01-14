from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score 

train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator = train_datagen.flow_from_directory(
        '../data/dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '../data/dataset/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)
test_generator = test_datagen.flow_from_directory(
        '../data/dataset/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)


# load model 
model = load_model("mask_detector_model_experimental.h5")

# Create Confusion Matrix
y_hat = model.predict(test_generator)
y_hat = np.argmax(y_hat, axis = 1)
y_true = test_generator.labels
classes = ['Mask', 'No Mask']
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_hat).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Create Classification Report
print(classification_report(y_true,y_hat,target_names=classes))

# Create ROC Curve
fpr, tpr, thresholds = roc_curve(y_true_val,y_hat_val) 
def plot_roc_curve(fpr,tpr):  
        plt.plot(fpr,tpr)  
        plt.axis([0,1,0,1])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.show()
