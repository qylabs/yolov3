import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3
import os
from PIL import Image

def representative_data_gen():
    path = '/home/allen/git/yolov3/mydataset/train_qy/images'  
    imgSet = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for name in imgSet:
        img = Image.open(name)
        img = img.convert('L')
        img = np.array(img.resize((160,128)))
        img = (img/255.0)
        img = np.array([img.astype('float32')])
        img = np.expand_dims(img, 0)
        yield [img]

path = '/home/allen/git/yolov3/runs/train/exp2/weights/qat_no_quant.pb'
converter = tf.lite.TFLiteConverter.from_saved_model(path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
# converter.inference_input_type = tf.float32
# converter.inference_output_type = tf.float32


tflite_model_quant = converter.convert()

import pathlib

tflite_models_dir = pathlib.Path("./runs/tflite_quant/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"qat_quant_static.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)

converter = tf.lite.TFLiteConverter.from_saved_model(path)
tflite_model_noquant = converter.convert()
tflite_model_noquant_file = tflite_models_dir/"qat_noquant.tflite"
tflite_model_noquant_file.write_bytes(tflite_model_noquant)

converter = tf.lite.TFLiteConverter.from_saved_model(path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"qat_quant_dynamic.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)