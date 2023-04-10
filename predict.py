import os
import sys
import argparse
import warnings

from PIL import Image

sys.path.append(os.path.join(os.getcwd(), 'src'))
import bottle

parser = argparse.ArgumentParser()

parser.add_argument('image', type = str, help = 'the image url to predict' )
parser.add_argument('-bottle-cap-model', dest='bottle_cap_model_file_name', type = str, help='file name of bottle cap model' )
parser.add_argument('-dest', dest='dest', type = str, default='./', help='destination folder' )
parser.add_argument('-threshold', dest='threshold', type = float, default=.5, help='destination folder' )
parser.add_argument( '-debug', type = bool, default = False, help='prints debug info in the image' )

args = parser.parse_args()

if not os.path.exists(args.image):
    print(f'image not exist {args.image}')
    exit()

if not os.path.exists(args.bottle_cap_model_file_name):
    print(f'model not exist {args.bottle_cap_model_file_name}')
    exit()
    
filename, file_extension = os.path.splitext(args.image)

warnings.filterwarnings('ignore')

device = bottle.utils.get_device()

bottle_detection_model = bottle.model.build_bottle_detection_model(device)
bottle_cap_state_model = bottle.model.build_bottle_cap_state_detection_model(device, False)

# switch to evaluation mode
bottle_detection_model.eval()
pred = bottle.image.get_prediction(bottle_detection_model, img_path=args.image, class_names=bottle.model.COCO_CLASS_NAMES)
masks, boxes, classes, scores = bottle.image.filter_prediction(pred, threshold=args.threshold, class_filter=['bottle'])

bottles_detected = len(masks)
images = []

if bottles_detected == 0:
    print('nothing detected')
    exit()
elif bottles_detected == 1: images = [Image.open(args.image)]
else: images = bottle.image.split_images_by_mask(args.image, masks, boxes)
        
bottle_cap_state_model_wrapper = bottle.model.ModelWrapper(bottle_cap_state_model)
bottle_cap_state_model_wrapper.load(args.bottle_cap_model_file_name)

predicted_cap_state = []
for i in range(len(images)):
    predicted_cap_state += [bottle_cap_state_model_wrapper.predict(images[i], labels = ('opened', 'closed'))]

new_classes = []
for i in range(len(classes)):
    new_classes += [f'{predicted_cap_state[i]} {classes[i]}']

if args.debug:
    img = bottle.image.plot_segments(img_path=args.image,
                                     pred_data=[masks, boxes, new_classes, scores],
                                     text_thickness=2,
                                     draw_score=False,
                                     draw_mask=False,
                                     mask_alpha=.3,
                                     text_size=1)
    bottle.image.save_image(img, f'{filename}-debug{file_extension}')

print(predicted_cap_state)