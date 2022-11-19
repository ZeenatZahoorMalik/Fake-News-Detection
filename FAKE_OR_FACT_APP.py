#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import base64

model = tf.keras.models.load_model("C:/Users/DELL/Downloads/Fake-News-Classification-master/demo/my_model.h5")
encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file("encoder")
st.title("FAKE OR FACT")
if "button_clicked" not in st.session_state:
  st.session_state.button_clicked = False
def callback():
  st.session_state.button_clicked = True

col13,col14 = st.columns(2)
with col13:
  
  if st.button('ABOUT THE WEBSITE'):
    st.markdown("Fake news can be very dangerous as it can spread misinformation and inflict rage in public. It is now becoming a serious problem in India due to more and more people using social media and lower levels of digital awareness. DEVELOPED BY -- **Zeenat Zahoor**.")
  if (
    st.button('CONNECT WITH US !', on_click=callback)
  or st.session_state):


    if st.button('E-MAIL'):
        st.markdown('🖄 zeenatzahoormalik@gmail.com')
    
    if st.button('LINKEDIN'):
      st.markdown('https://www.linkedin.com/in/zeenat-zahoor-271121238/')
    
with col14:
    st.image("https://miro.medium.com/max/1400/1*BQ5j-HiRrcc0eIqhqJYtoQ.gif")
    
#Add image using URL 
def add_bg_from_url():
    
    st.markdown(
          f"""
          <style>
          .stApp {{
              background-image: url("https://image.shutterstock.com/image-illustration/fake-vs-fact-cube-black-260nw-1719242434.jpg");
              background-attachment: fixed;
              background-size: cover;

          }}
          </style>
          """,
          unsafe_allow_html=True
      )

add_bg_from_url()
def pad_to_size(vec, size):
  zero = [0] * (size - len(vec))
  vec.extend(zero)
  return vec


def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)


input_article = st.text_area("Enter article for checking")

 
   
if st.button("Submit"):
    predictions = sample_predict(input_article, pad=False)
    if predictions > 0.5 :
        st.text("FACT")
    else:
        st.text("FAKE")



# In[ ]:




