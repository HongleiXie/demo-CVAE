import numpy as np
import os
import sys
import tensorflow as tf
from model import CVAE
import matplotlib.pyplot as plt
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model(PATH):
    model = CVAE(2, 128)
    # Restore the weights
    model.load_weights(PATH)
    return model

def main():
    st.title('Generate Synthetic Handwriting Digits')
    st.sidebar.title('MNIST')
    digit = st.sidebar.selectbox('Pick a digit from 0~9', range(0,10))
    num_examples_to_generate = st.sidebar.selectbox('Pick the number of generated images', (4, 8, 16))

    model = load_model('saved_model/my_checkpoint')
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, model.latent_dim])
    y = np.zeros(shape=(10))
    np.put(y, digit, 1)
    y = np.array(num_examples_to_generate*[y])
    pred = model.sample(random_vector_for_generation, tf.convert_to_tensor(y))

    all_y = [pred[i, :, :, 0].numpy() for i in range(num_examples_to_generate)]
    for i in range(int(num_examples_to_generate / 4)):
        st.image(image=all_y[i*4 : (i+1)*4], width=64)

    st.markdown('[Project Page](https://github.com/HongleiXie/demo-CVAE)')

if __name__ == '__main__':
    main()
