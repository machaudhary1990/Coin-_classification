# %%
"""
# Imports
"""

# %%
import tensorflow as tf
import cv2,os
from tensorflow.keras import layers
import numpy as np
#from tqdm import tqdm
from datetime import datetime
class Classify_NET:
    def __init__(self,image_size=(640,480)):
        self.image_size = image_size+(3,)
        
    
    def get_model(self):
        inputs = tf.keras.Input(shape=self.image_size)

        #Encoding image 1
        incep = tf.keras.applications.MobileNet(include_top=False)
        x1 = incep(inputs)
        x1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)

        #Decoding part
        x1 = tf.keras.layers.Concatenate()([x1,x2])
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Dense(1024,activation="relu")(x1)
        x1 = tf.keras.layers.Dense(128,activation="relu")(x1)
        x1 = tf.keras.layers.Dense(1,activation="relu")(x1)
        
        model = tf.keras.Model([inputs,x1)
        return model
if __name__ == "__main__":
    # %%
    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()
    from database import DataGenerator
    # Build model
    target_size = (480,480)
    idnet = ID_NET(image_size =target_size)
    model = idnet.get_model()
    model.summary()
    #model.compile(optimizer='adam', loss='categorical_crossentropy')
    training_generator = DataGenerator("./dataset",target_size=target_size,batch_size=2,shuffle=False)
    if not os.path.exists('checkpoints_dnn'):
        os.makedirs('checkpoints_dnn') 
    checkpoint_path = "checkpoints_dnn/weights.{epoch:02d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weighs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    logdir = "logs_dnn/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logs = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(training_generator,epochs=100,callbacks=[cp_callback,tb_logs],verbose = 1)
