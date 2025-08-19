import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras import layers, Model


st.markdown(
    """
    <style>
    .stApp {
        background-color: #C39898;  
    }

    /* Change color of the top navbar */
    header[data-testid="stHeader"] {
        background-color: #987070;  
    }

    /* Change color of the sidebar */
    section[data-testid="stSidebar"] {
        background-color: #D29F80;  

    /* Custom Font Colors */


    /* Custom Subheader */
    .subheader {
        color: #00FF00; /* Green */
        font-size: 25px;
        font-weight: 500;
    }

    /* Custom text */
    .custom-text {
        font-size: 18px;
        color: #6F4E37; 
    }


    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h1 style="text-align: center; color: #6F4E37;">WingSpotter – Bird Species Identifying Platform Classifier</h1>',
    unsafe_allow_html=True
)
# Define the model architecture
pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='max'
)

pretrained_model.trainable = False


# Load the trained model and labels

# Add your custom head
inputs = pretrained_model.input
x = layers.Dense(128, activation='relu')(pretrained_model.output)
x = layers.Dropout(0.45)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.45)(x)
outputs = layers.Dense(525, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model (make sure to use the same optimizer, loss, and metrics as during training)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load the weights
model.load_weights("model_weights.h5")

class_indices = np.load("labels.npy", allow_pickle=True).item()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    fixed_size = (600, 500)
    #display_size = (600, 200)  # Adjusted size
    image_resized = image.resize(fixed_size)

    st.markdown(
        """
        <style>
        .centered-image {
            display: flex;
            justify-content: center;
        }
        </style>
        <div class="centered-image">
        """,
        unsafe_allow_html=True,
    )

    st.image(image_resized, caption="Uploaded Image", width=300)  # Adjust width as needed

    st.markdown("</div>", unsafe_allow_html=True)

    #st.image(image, caption="Uploaded Image", width=200)
    #st.markdown('<p class="uploaded-text">Image uploaded successfully!</p>', unsafe_allow_html=True)

    #st.image(image, caption="Uploaded Image", use_column_width=True)
    
    IMAGE_SIZE = (224, 224)  
    if image.size != IMAGE_SIZE:
        image = image.resize(IMAGE_SIZE)


    # Preprocess the image
    #img = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_indices[predicted_class_index]

    print("Prediction:", predicted_class_label)
    st.header("Prediction:")
    st.subheader(predicted_class_label)

