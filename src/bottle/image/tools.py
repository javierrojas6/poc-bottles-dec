import numpy as np
import random
import cv2
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

def colored_mask(mask):
  """
  The function takes a binary mask and returns a colored mask by assigning random colors to the pixels
  with value 1.
  
  :param mask: The input mask is a binary image where the object of interest is represented by white
  pixels (pixel value of 1) and the background is represented by black pixels (pixel value of 0). The
  function `colored_mask` takes this binary mask as input and returns a colored mask where the object
  :return: The function `colored_mask` returns a numpy array of shape `(height, width, 3)` where each
  pixel in the array is colored according to the input `mask`. The color of each pixel is randomly
  chosen from a list of predefined colors `COLORS`.
  """
  r = np.zeros_like(mask).astype(np.uint8)
  g = np.zeros_like(mask).astype(np.uint8)
  b = np.zeros_like(mask).astype(np.uint8)

  r[mask == 1], g[mask == 1], b[mask == 1] = COLORS[random.randrange(0, len(COLORS))]
  return np.stack([r, g, b], axis = 2)

def get_prediction(model, img_path, class_names, transforms = []):
  """
  This function takes in an image, a model, a list of class names, and optional transforms, and
  returns the predicted masks, boxes, classes, and scores for objects in the image using the model.
  
  :param model: The object detection model used for making predictions on the input image
  :param img_path: The file path of the image to be used for prediction
  :param class_names: a list of names for the different classes that the model can predict
  :param transforms: A list of image transformations to be applied to the input image before passing
  it to the model for prediction. If no transformations are provided, a default transformation of
  converting the image to a tensor is applied
  :return: The function `get_prediction` returns four arrays: `masks`, `boxes`, `classes`, and
  `scores`. These arrays contain the predicted segmentation masks, bounding boxes, class labels, and
  confidence scores for the input image using the specified `model` and `class_names`.
  """
  img = Image.open(img_path)

  if len(transforms) == 0:
    transforms = T.Compose([T.ToTensor()])

  img = transforms(img)
  pred = model([img])

  scores = np.array(list(pred[0]['scores'].detach().numpy()))
  masks = np.array((pred[0]['masks'] > 0).squeeze().detach().cpu().numpy())
  classes = np.array([class_names[i] for i in list(pred[0]['labels'].detach().numpy())])
  boxes = np.array([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())])

  return masks, boxes, classes, scores

def filter_prediction(prediction_data, threshold = .5, class_filter = []):
  """
  This function filters prediction data based on a threshold and optional class filter.
  
  :param prediction_data: a tuple containing the predicted masks, boxes, classes, and scores for an
  object detection task
  :param threshold: The minimum score threshold for a prediction to be considered valid. Any
  predictions with a score below this threshold will be filtered out
  :param class_filter: A list of classes to filter the predictions by. If this parameter is not empty,
  only predictions with classes in this list will be returned
  :return: The function `filter_prediction` returns the filtered masks, boxes, classes, and scores
  based on the given threshold and class filter.
  """
  masks, boxes, classes, scores = prediction_data

  condition = scores >= threshold

  if len(class_filter) > 0:
    condition &= np.in1d(classes, class_filter)

  masks = masks[condition]
  boxes = boxes[condition]
  classes = classes[condition]            
  scores = scores[condition] 

  return masks, boxes, classes, scores

def plot_segments(img_path,
                  pred_data,
                  draw_bounding_boxes = True,
                  draw_labels = True,
                  draw_score = True,
                  draw_mask = True,
                  mask_alpha = .5,
                  score_fix = 2,
                  rect_thickness = 1,
                  text_size = 1,
                  text_thickness = 2):
  """
  The function plots segments on an image with optional bounding boxes, labels, and scores.
  
  :param img_path: The file path of the image to be plotted
  :param pred_data: A tuple containing the predicted masks, bounding boxes, predicted classes, and
  scores for each object in the image
  :param draw_bounding_boxes: A boolean parameter that determines whether or not to draw bounding
  boxes around the detected segments. If set to True, bounding boxes will be drawn. If set to False,
  bounding boxes will not be drawn, defaults to True (optional)
  :param draw_labels: A boolean parameter that determines whether or not to draw the predicted class
  labels on the image, defaults to True (optional)
  :param draw_score: A boolean parameter that determines whether or not to draw the score of the
  predicted object on the image, defaults to True (optional)
  :param score_fix: The number of decimal places to round the score to when displaying it on the
  image, defaults to 2 (optional)
  :param rect_thickness: The thickness of the bounding box rectangle to be drawn around the detected
  object, defaults to 1 (optional)
  :param text_size: The size of the text used for the labels, defaults to 1 (optional)
  :param text_thickness: The thickness of the text in the plotted image, defaults to 2 (optional)
  :return: an image with plotted segments, bounding boxes, labels, and scores based on the input image
  path and predicted data.
  """
  
  masks, boxes, pred_cls, scores = pred_data
  img = cv2.imread(img_path)

  if draw_score:
    format_string = "{:.Xf}".replace('X', str(score_fix))

  for i in range(len(masks)):
    if draw_mask:
      rgb_mask = colored_mask(masks[i])
      img = cv2.addWeighted(img, 1, rgb_mask, mask_alpha, 0)

    if draw_bounding_boxes:
      cv2.rectangle(
          img,
          (int(boxes[i][0][0]), int(boxes[i][0][1])),
          (int(boxes[i][1][0]), int(boxes[i][1][1])),
          color=(0,255,0),
          thickness=rect_thickness
      )

    if draw_labels:
      label_params = [pred_cls[i]]

      if draw_score:
        label_params += [format_string.format(scores[i])]

      cv2.putText(
          img,
          " ".join(label_params),
          (int(boxes[i][0][0]) + 5, int(boxes[i][0][1]) + 20),
          cv2.FONT_HERSHEY_SIMPLEX,
          text_size,
          (0, 255, 0),
          thickness=text_thickness
      )

  return img

def save_image(img, filename):
  return cv2.imwrite(filename, img)

def split_images_by_mask(img, masks, boxes):
  """
  This function takes an image, masks, and boxes as inputs, and returns a list of images that have
  been split based on the masks and boxes.
  
  :param img: The path to the image file that needs to be split
  :param masks: A list of binary masks where each mask corresponds to an object in the image
  :param boxes: The "boxes" parameter is a list of bounding boxes, where each bounding box is
  represented as a list of two points. The first point represents the top-left corner of the bounding
  box, and the second point represents the bottom-right corner of the bounding box
  :return: a list of images that have been split from the original image based on the provided masks
  and boxes.
  """
  img = Image.open(img)
  
  imgs = []
  for i in range(len(masks)):
    copied_image = np.copy(img)
    int_mask = masks[i].astype(int)
    copied_image[int_mask == 0] = [0, 0, 0]
    
    start_x = int(boxes[i][0][0])
    start_y = int(boxes[i][0][1])

    end_x = int(boxes[i][1][0])
    end_y = int(boxes[i][1][1])
    
    imgs += [copied_image[
      start_y: start_y + end_y,
      start_x: start_x + end_x
      ]
    ]
    
  return imgs

def transform_resize_to_max_size(img_path, max):
    img = Image.open(img_path)
    img_shape =  np.array(img).shape
    image_max = np.max(img_shape)
    
    if image_max <= max: # nothing to do
        return
      
    w, h, _ = img_shape
    cr =  max / image_max
    
    w, h = int(w * cr), int(h * cr)
    transform_resize = T.Compose([
        T.ToTensor(),
        T.Resize((w, h)),
        T.ToPILImage()
    ])
    
    # get normalized image
    img_normalized = transform_resize(img)
    img_normalized.save(img_path)