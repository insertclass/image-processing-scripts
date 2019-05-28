# USAGE
# python detect_image.py --image test-data/image1.jpg -tpath examples

from imageai.Detection import ObjectDetection
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-t", "--tpath", required=True,
	help="path to target image folder")
args = vars(ap.parse_args())

execution_path = os.path.dirname(os.path.realpath(__file__))

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom = detector.CustomObjects(person=True, bird=True, bicycle=True, cat=True, airplane=True,
                                boat=True, bottle=True, chair=True, dining_table=True, potted_plant=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=args['image'],
                                             output_image_path=args['tpath'],
                                             minimum_percentage_probability=50)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")