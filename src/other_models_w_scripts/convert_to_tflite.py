import tensorflow as tf
from tensorflow.keras.models import load_model

# load model
# model = load_model('model_w_old_data/mask_detector_model.h5')
model = load_model('mask_detector_model_new.h5')

# instantiate representative training data generator for post-quantization training
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files("../data/dataset/all_images" + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

#quantize and convert keras model to edge tpu-compatible tflite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()


# save tflite model
with open('mask_detector_model_new.tflite', 'wb') as f:
  f.write(tflite_model)