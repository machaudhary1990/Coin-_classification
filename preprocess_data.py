import os
import cv2
import argparser 


def crop_images(inputfile, out_dir):
    try:
        img = cv2.imread(inputfile)
        json_file = inputfile.replace(".jpg",".json")
        json_data = None
        
        with open(json_file,'r'):
        	json_data = json.load(f)

        for shape in json_data['shapes']:
        	minx,miny,maxx,maxy= None,None,None,None
        	for pt in shape['points']:
        	
        	minx = pt[0] if (minx==None or pt[0] < minx) else minx 		
        	miny = pt[0] if (miny==None or pt[0] < miny) else miny
        	maxx = pt[0] if (maxx==None or pt[0] > maxx) else maxx
        	maxy = pt[0] if (maxy==None or pt[0] < maxy) else maxy

        	new_image = img[miny:maxy, minx:maxx]
            f_name = str(os.path.basename(inputfile)).replace(".", "_{}.".format(i))
            cv2.imwrite("{}/{}".format(out_dir, f_name), new_image)
    except:
        print(filename)




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Cropping images.')
    parser.add_argument('--input_dir', required=True, help='Input dir of the dataset')
    parser.add_argument('--output_dir', required=True, help='Output dir to be generated')
    parser.add_argument('--max_threads', type=int, default=15, help='Maximum threads allowed')
    args = parser.parse_args()
    max_threads = args.max_threads
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jpg") and td.active_count() >= max_threads:
            while True:
                time.sleep(1)
                if td.active_count() < max_threads:
                    break
        thread = td.Thread(target=crop_images, args=[os.path.join(args.input_dir, filename), args.output_dir],
                           daemon=True)
        thread.start()