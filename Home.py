import cv2
import numpy as np
from matplotlib import pyplot
import streamlit as st
import random


st.set_page_config(
    page_title='CNN: Alphabet and Number Recognition',
    page_icon='🧠'
)


def random_image():
    st.session_state.image = 'test_image_' + str(random.randint(1, 6)) + '.jpg' 
    return


def main():
    st.write('''
        # Alphabet and Number Recognition using CNN Model
    ''')
    st.write('''
        ### Step 1: Input image
        Please choose either to use uploaded image or to take photo using webcamera.
    ''')

    error_message = 'Due to techniqal difficulty, this feature is not working correctly. A sample image is used in this case.'

    with st.expander('A. Upload image file'):
        st.file_uploader(error_message, type=['png', 'jpg', 'jpeg'], disabled=True)

    with st.expander('B. Use camera to take photo'):
        st.camera_input(error_message, disabled=True)

    st.button('Randomize Test Image', on_click=random_image)

    st.write('''
        ### Step 2: Detect any potential characters using OpenCV
    ''')

    if 'image' not in st.session_state:
        st.session_state.image = 'test_image_1.jpg'

    st.image(st.session_state.image, st.session_state.image, width=400)

    # Load the image using OpenCV
    image = cv2.imread(st.session_state.image)

    # Extract image dimensions
    height, width, _ = image.shape
    char_recognized = 0

    # Resize the image for better processing
    image = cv2.resize(image, dsize=(width * 5, height * 4), interpolation=cv2.INTER_CUBIC)

    # Convert image to grayscale for easier segmentation
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Dilate the image to accentuate character regions
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Apply Gaussian blur to reduce noise and smooth edges
    gsblur = cv2.GaussianBlur(img_dilation, (5, 5), 0)

    # Find contours (potential character regions)
    ctrs, _ = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right for character recognition
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Create a copy of the original image for displaying processed results
    processed_image = image.copy()

    for _, ctr in enumerate(sorted_ctrs):
        # Create bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        color = (0, 255, 0)

        # Assuming image width and height that are less than 64px is just noises
        if (w < 64 and h < 64):
            color = (255, 0, 0)  # Mark it in red color

        cv2.rectangle(processed_image, (x - 10, y - 10), ( x + w + 10, y + h + 10 ), color, 10)
        char_recognized += 1

    st.image(processed_image, 'Possible character recognized is ' + str(char_recognized) , width=400)
    st.write('Notice that rectangles in red are the potential noises, it is detected if any image is below the setting, which is 24 x 24 px.' \
             ' While, the rectangles in green are the potential characters detected.')

    st.write('### Step 3: Remove noises')

    character_images = []
    char_recognized = 0

    for _, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        if not (w < 64 and h < 64):
            word_image = gray[y:y+h, x:x+w]
            resized_image = cv2.resize(word_image, (28, 28), interpolation=cv2.INTER_AREA)
            character_images.append(resized_image)  # Add the image to the list
            # print(str(i + 1) + '. ' + str(word_image.shape))  # Debug
            char_recognized += 1

    combined_image = np.concatenate(character_images, axis=1)

    st.image(combined_image, 'Possible character recognized is ' + str(char_recognized))
    st.write('Now the previous noises has been removed, and the number of possible character recognized are now reduced and separated for the next process.')

    st.write('''
        ### Step 4: Load trained model
        ''')
    uploaded_model = st.file_uploader('Upload the trained model file (model.h5):', type=['h5'])
    
    # from tensorflow.keras.models import load_model
    #
    # if st.session_state.model is None:
    #    st.session_state.model = 'model.h5'
    #
    # st.session_state.model = uploaded_model
    #
    # model = load_model(st.session_state.model)

    st.write('''
        ### Step 5: Compare with the trained model *
        This is the stage where each characters detected is compared with the trained EMNIST model.
    ''')


main()
