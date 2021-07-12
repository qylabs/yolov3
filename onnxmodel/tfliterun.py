import tensorflow as tf
import numpy as np

def tfliterun(model_path,input_array):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('Input details: ', input_details)
    print('Output details: ', output_details)
    scale, zero_point = input_details[0]['quantization']

    
    interpreter.set_tensor(input_details[0]['index'], img.reshape(input_details[0]['shape']))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    return output

if __name__=="__main__":
    from PIL import Image
    img_path=''
    img = np.array(Image.open(img_path))
    print('img.shape ',img.shape)