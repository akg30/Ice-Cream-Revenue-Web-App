# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:40:04 2023

@author: Hp
"""

import streamlit as st
import numpy as np
import pickle as pk

loaded_model=pk.load(open('trained_model.sav','rb'))

def revenue(input_temperature):
    input_temperature_array=np.asarray(input_temperature)
    input_temperature_array_new=np.reshape(input_temperature_array, (1,-1))
    revenue_predict=loaded_model.predict(input_temperature_array_new)
    import math 
    return 'The Revenue Based On Temperature {1} Celcius is {0} Rupees'.format(math.floor(revenue_predict),input_temperature)

def main():
    st.title('Ice Cream Sales Revenue Prediction Using Machine Learning')
    temp=st.number_input('Enter Temperature')
    revenue_show=' '
    if st.button('Click Here To Get Revenue'):
        revenue_show=revenue(temp)
    st.success(revenue_show)
    st.subheader('Exploratory Data Analysis Done And Machine Learning Model Deployed By "Anubhav Kumar Gupta"')

if __name__=="__main__":
    main()
              
