import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('/Users/aishwaryapalta/my-react-app/src/note_identifier_model.h5')
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

label_map = {0: '10' , 1: '10' , 2: '100',3: '100',4: '20', 5: '20',6:'200',7:'50',8:'50',9:'500'}  

def identify_note_denomination(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class

# Example usage
img_path = '/Users/aishwaryapalta/Documents/Denominations 2/50old/50_test1.jpeg'
img2_path = '/Users/aishwaryapalta/Documents/20Note.jpeg'
denomination = identify_note_denomination(img_path)
print(f'The denomination of the note is: {denomination}')