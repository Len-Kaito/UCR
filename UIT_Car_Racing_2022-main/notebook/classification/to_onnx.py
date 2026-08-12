from tensorflow.keras.models import load_model
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx




model = load_model('models/model-016.h5')
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model,'my_sign_none.onnx')
