import onnx
import onnx_tf
import tensorflow as tf

def onnx2pb(onnx_model_path):
    try:
        onnx_model = onnx.load(onnx_model_path)  # load onnx model
        from onnx_tf.backend import prepare
        tf_exp = prepare(onnx_model)  # prepare tf representation

        pb_model_path=onnx_model_path.replace('onnx','pb')
        tf_exp.export_graph(pb_model_path)  # export the model
        print('saved_model export success, saved as {}'.format(pb_model_path) )

        return pb_model_path
        
    except Exception as e:
        print('saved_model export failure: %s' % e)
    

def pb2tflite(pb_model_path):
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_model_path)
        # tell converter which type of optimization techniques to use
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # to view the best option for optimization read documentation of tflite about optimization
        # to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
        # convert the model
        tflite_model = converter.convert()

        tflite_model_path=pb_model_path.replace('pb','tflite')
        with open(tflite_model_path,'wb') as g:
            g.write(tflite_model)
        print('tflite export success, saved as {}'.format(tflite_model_path) )

        return tflite_model_path

    except Exception as e:
        print('tflite export failure: %s' % e)


if __name__=="__main__":
    onnx_model_path='weights/yolov3-tiny_sim_relu.onnx'
    pb_model_path=onnx2pb(onnx_model_path)
    print(pb_model_path)
    tf_model_path=pb2tflite(pb_model_path)
    print(tf_model_path)