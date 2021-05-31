import os
import cv2
import threading as td
import argparse
import math
import time
import json
def crop_images(inputfile, out_dir):
	try:
		img = cv2.imread(inputfile)    	
		json_file = inputfile.replace(".jpg",".json")    	
		json_data = None    	   	
		f = open(json_file,"r")        
		json_data = json.load(f)
		h,w,c = img.shape
		for i,shape in enumerate(json_data['shapes']):            
			minx,miny,maxx,maxy= None,None,None,None 
			x1,y1 = int(shape['points'][0][0]),int(shape['points'][0][1])
			x2,y2 = int(shape['points'][1][0]),int(shape['points'][1][1])      
			radius = math.sqrt((x2-x1)**2+(y2-y1)**2)		        
			minx = int(x1-radius) if (x1-radius)> 0 else 0
			miny = int(y1-radius) if (y1-radius)> 0 else 0
			maxx = int(x1+radius) if (x1+radius)< w else w
			maxy = int(y1+radius) if (y1+radius)< h else h 
			new_image = img[miny:maxy, minx:maxx]                
			f_name = str(os.path.basename(inputfile)).replace(".", "_{}.".format(i))                
			cv2.imwrite("{}/{}".format(out_dir, f_name), new_image)    
	except Exception as e:  
		# pass      
		# print("Exception in ",filename)        
		print(e)

if __name__ == "__main__":    
	parser = argparse.ArgumentParser(description='Cropping images.')    
	parser.add_argument('--input_dir', required=True, help='Input dir of the dataset')    
	parser.add_argument('--output_dir', required=True, help='Output dir to be generated')    
	parser.add_argument('--max_threads', type=int, default=15, help='Maximum threads allowed')    
	args = parser.parse_args()    
	max_threads = args.max_threads    
	for filename in os.listdir(args.input_dir):        
		if filename.endswith(".jpg") and td.active_count() <= max_threads:            
			while True:                
				time.sleep(1)                
				if td.active_count() < max_threads:                    
					break            
			thread = td.Thread(target=crop_images, args=[os.path.join(args.input_dir, filename), args.output_dir],daemon=True)            
			thread.start()