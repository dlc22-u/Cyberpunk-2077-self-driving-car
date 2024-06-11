import os
import glob
import sys
import cv2
import numpy as np
from keras.models import load_model

# try:
target_folder = sys.argv[1] + "/text"
all_image_paths = glob.glob(os.path.join(target_folder, '*.png'))

# Extract the numeric part of the filename and use it as the sorting key
all_image_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

print(all_image_paths)

speed_model = load_model('speed_rec.h5')

predictions = []
for image_path in all_image_paths:
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension if needed
    prediction = speed_model.predict(image)
    predicted = np.argmax(prediction, axis=1)[0]
    print(predicted)
    predictions.append(predicted)

print(predictions)

# Check if speed.txt exists, if not, create it
output_file_path = os.path.join(target_folder, 'speed.txt')
if not os.path.exists(output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(" ")
    with open(output_file_path, 'w') as f:
        f.write('\n'.join(map(str, predictions)))
else:
    print("file exist. exit")
# except:
#     print("인자값 오류")