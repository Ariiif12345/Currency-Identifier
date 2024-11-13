import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import joblib
import os

# Load the Random Forest model from the .pkl file
model_filename = os.path.join(os.getcwd(), 'random_forest_note_identifier.pkl')
rf_model = joblib.load(model_filename)

# Load the pre-trained VGG16 model + higher level layers for feature extraction
local_weights_file = '/Users/aishwaryapalta/my-react-app/src/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = tf.keras.applications.VGG16(weights=local_weights_file, include_top=False, input_shape=(150, 150, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=tf.keras.layers.Flatten()(base_model.output))

label_map = {0: '10', 1: '20', 2: '50', 3: '100', 4: '200', 5: '500'}

def identify_note_denomination(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Extract features using the VGG16 model
    features = model.predict(img_array)

    # Predict using the Random Forest model
    prediction = rf_model.predict(features)
    predicted_class = prediction[0]

    return predicted_class

# Example usage
img_path = '/Users/aishwaryapalta/Documents/Denominations 2/200/200_test.jpeg'
img2_path = '/Users/aishwaryapalta/Documents/200Note.jpeg'
denomination = identify_note_denomination(img_path)
print(f'The denomination of the note is: {denomination}')