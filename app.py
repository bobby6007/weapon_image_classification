#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:38:43 2022

@author: boburjonbahtiyorov
"""

import streamlit as st 
from fastai.vision.all import *
import plotly.express as px
import pathlib






plt=platform.system()
temp = pathlib.PosixPath
pathlib.PosixPath

#title
st.title('Weapons Classifier')

# image uploading
file = st.file_uploader('Upload a picture here', type=['png','jpeg','gif','svg'])
if file:
        st.image(file)
        # PIL conver
        img = PILImage.create(file)
        
        # model
        model = load_learner('weapon_model.pkl','rb')
        
        # prediction
        pred, pred_id, probs=model.predict(img)
        st.success(f'Prediction:{pred}')
        st.info(f'Probability {probs[pred_id]*100:.1f}%')
        
        
        # plotting 
        fig = px.bar(x=probs*100, y=model.dls.vocab)
        st.plotly_chart(fig)
