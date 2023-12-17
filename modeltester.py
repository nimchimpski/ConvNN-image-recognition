import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import sys
import PIL
from traffic import IMG_HEIGHT, IMG_WIDTH, NUM_CATEGORIES
import cv2

# Check command-line arguments
if len(sys.argv) != 3:
    sys.exit("Usage: python3 modeltester.py model testimage")

# Load  model 
model = keras.models.load_model(sys.argv[1])

# Load and preprocess a test image using openCV
test_image_path = sys.argv[2]
print(f"---type test image path={type(test_image_path)}")
img = cv2.imread(test_image_path)
print(f"---type img={type(img)}")
img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
# Normalize pixel values to [0, 1]
img/255.0 
# Add batch dimension
img_array = np.expand_dims(img, axis=0)  
# print number of dimensions
print(f"---number of dimensions={img_array.ndim}")





# Make predictions
predictions = model.predict(img_array)

# Interpret the predictions 
# Get the index of the class with the highest probability
class_index = np.argmax(predictions)  
class_names = ["stop" , "left", "20"]
# Map the index to a class label
class_label = class_names[class_index]  
print(f'The predicted class is: {class_label}')
