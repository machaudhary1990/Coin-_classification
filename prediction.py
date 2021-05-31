import tensorflow as tf
from Coin_classification import Classify_NET
class Coin_Prediction:
	def __init__(self,model_path,target_size=(480,480)):
		self.target_size = (480,480)
		self.labels = [5, 10, 25, 50, 100]
    	self.classify = Classify_NET(image_size =target_size)
    	self.model = classify.get_model()
    	self.model.summary()
    	self.model.load(model_path)
    	
    def predict(self,image):
    	X=[]
    	in_img = cv2.resize(image,self.target_size)
    	in_img = np.transpose(in_img,(1,0,2))
    	X.append(in_img)
    	X = np.array(X)
    	predictions = self.model.predict(batch)
        pred =  predictions[0]
        pred = tf.nn.softmax(pred)
       	result = max(pred)
       	idx = pred.index(result)
       	return self.labels[idx]


if __name__=="__main__":
	
	import cv2
	import numpy as np

	coin_predictor = Coin_Prediction("MODEL_CHECKPOINT_PATH")
	image = cv2.imread("INPUT_IMAGE")
	output = coin_predictor.predict(image)
	print(output)