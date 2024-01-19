import streamlit as st


st.set_page_config(
    page_title='Train EMNIST with Keras',
    page_icon='🧠'
)

st.write('# Train EMNIST using Keras on Kaggle')
st.write('### Step 1: Import modules')
st.code(
    '''
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.utils import np_utils
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    '''
)

st.write('### Step 2: Import datasets')
st.code('''
    train = pd.read_csv("../input/emnist-balanced-train.csv",delimiter = ',')
    test = pd.read_csv("../input/emnist-balanced-test.csv", delimiter = ',')
    mapp = pd.read_csv("../input/emnist-balanced-mapping.txt", delimiter = ' ', \
                    index_col=0, header=None, squeeze=True)
    print("Train: %s, Test: %s, Map: %s" %(train.shape, test.shape, mapp.shape))
    '''
)

st.write('### Step 3: Split x and y')
st.code(
    '''
    train_x = train.iloc[:,1:]
    train_y = train.iloc[:,0]
    del train

    test_x = test.iloc[:,1:]
    test_y = test.iloc[:,0]
    del test

    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    '''
)

st.write('### Step 4: Flip, rotate, and normalize images for CNN')
st.code(
    '''
    # Define a function to rotate an image 90 degrees clockwise and flip it horizontally
    def rotate(image):
        image = image.reshape([HEIGHT, WIDTH])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    # Convert training data to NumPy array for efficient operations
    train_x = np.asarray(train_x)

    # Apply the rotate function to each image in the training set
    train_x = np.apply_along_axis(rotate, 1, train_x)
    print ("train_x:",train_x.shape)

    # Similar operations for test data
    test_x = np.asarray(test_x)
    test_x = np.apply_along_axis(rotate, 1, test_x)
    print ("test_x:",test_x.shape)

    # Normalize pixel values to the range [0, 1]
    train_x = train_x.astype('float32')  # Ensure float values for division
    train_x /= 255  # Divide by 255 for normalization
    test_x = test_x.astype('float32')
    test_x /= 255

    # One-hot encode labels for categorical classification
    num_classes = train_y.nunique()  # Count the number of unique classes
    train_y = np_utils.to_categorical(train_y, num_classes)
    test_y = np_utils.to_categorical(test_y, num_classes)

    # Print the shapes of the encoded labels
    print("train_y: ", train_y.shape)
    print("test_y: ", test_y.shape)

    # Reshape image for CNN
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)
    '''
)

st.write('### Step 5: Build model')
st.code(
    '''
    # ((Si - Fi + 2P)/S) + 1
    model = Sequential()

    model.add(Conv2D(filters=128, kernel_size=(5,5), padding = 'same', activation='relu',\
                    input_shape=(HEIGHT, WIDTH,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3) , padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''
)

st.write('### Step 6: Train model')
st.code(
    '''
    model.fit(train_x, train_y, epochs=10, batch_size=512, verbose=1, validation_data=(val_x, val_y))
    '''
)

st.write('### Step 7: Save model')
st.code(
    '''
    model.save('model.h5')
    '''
)