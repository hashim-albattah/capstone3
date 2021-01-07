import tensorflow as tf
from tensorflow.keras.models import load_model
# load and convert mask detector (keras) models to tflite
# model = load_model('mask_detector_model_new.h5')
model = load_model('mask_detector_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save tflite model
with open('mask_detector_model.tflite', 'wb') as f:
  f.write(tflite_model)