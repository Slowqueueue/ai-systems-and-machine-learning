import streamlit as st
import json
import pickle
import sklearn
import pandas as pd
import tensorflow as tf

cut_to_num = {
    "Fair":0,
    "Good":1,
    "Very Good":2,
    "Premium":3,
    "Ideal":4
}

color_to_num = {
    'J':0,
    'I':1,
    'H':2,
    'G':3,
    'F':4,
    'E':5,
    'D':6
}

clarity_to_num = {
    "I1":0,
    "SI2":1,
    "SI1":2,
    "VS2":3,
    "VS1":4,
    "VVS2":5,
    "VVS1":6,
    "IF":7
}

st.sidebar.header("Enter values:")

x = st.sidebar.slider('X:', min_value = 0.0, max_value = 10.78, value = 0.0, step = 0.01)

y = st.sidebar.slider('Y:', min_value = 0.0, max_value = 58.9, value = 0.0, step = 0.01)

z = st.sidebar.slider('Z:', min_value = 0.0, max_value = 31.8, value = 0.0, step = 0.01)

cut = st.sidebar.selectbox('Cut:', ("Fair", "Good", "Very Good", "Premium", "Ideal"))

color= st.sidebar.selectbox('Color:', ('J', 'I', 'H', 'G', 'F', 'E', 'D'))

clarity = st.sidebar.selectbox('Clarity:', ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"))

depth = st.sidebar.slider('Depth:', min_value = 43.0, max_value = 79.0, value = 43.0, step = 0.1)

carat = st.sidebar.slider('Carat:', min_value = 0.2, max_value = 5.01, value = 0.2, step = 0.01)

table = st.sidebar.slider('Table:', min_value = 43.0, max_value = 95.0, value = 43.0, step = 0.1)

df = pd.DataFrame([[x, y, z, cut_to_num[cut], color_to_num[color], clarity_to_num[clarity], depth, carat, table]])
df.columns = (['x','y','z','cut','color','clarity','depth','carat','table'])

st.write("Input values in the initial scale")
st.write(df)

cl1, cl2 = st.columns(2)

m1_file = open(r"C:\Users\Nikita\Desktop\siimorgrpython\m1.json", "r")
m1_dist = json.load(m1_file)
m1_file.close()

m2_file = open(r"C:\Users\Nikita\Desktop\siimorgrpython\m2.json", "r")
m2_dist = json.load(m2_file)
m2_file.close()

def Predict(dictt, values):
    mN = dictt["modelName"]
    r2 = dictt["R2"]
    rmse = dictt["RMSE"]

    if (mN == "m2.dump"):
        m2 = tf.keras.models.load_model(r"C:\Users\Nikita\Desktop\siimorgrpython\m2.keras")

        snx_file = open(r"C:\Users\Nikita\Desktop\siimorgrpython\scalerNormForX.dump", "rb")
        snx = pickle.load(snx_file)
        snx_file.close()

        sny_file = open(r"C:\Users\Nikita\Desktop\siimorgrpython\scalerNormForY.dump", "rb")
        sny = pickle.load(sny_file)
        sny_file.close()

        cl2.subheader("Neural network")
        cl2.write("R2 = " + str(r2))
        cl2.write("RMSE = " + str(rmse))
        cl2.write("Output data in normalized form:")

        predm2 = pd.DataFrame(m2.predict(snx.transform([values])))
        predm2.columns = (['price'])
        cl2.write(predm2)
        cl2.write("Output data in the initial scale:")
        predm2init = pd.DataFrame(sny.inverse_transform([[predm2.iloc[0, 0]]]))
        predm2init.columns = (['price'])
        cl2.write(predm2init)
    else:
        m1_out = open(r"C:\Users\Nikita\Desktop\siimorgrpython\m1.dump", "rb")
        m1 = pickle.load(m1_out)
        m1_out.close()

        cl1.subheader("Linear regression")
        cl1.write("R2 = " + str(r2))
        cl1.write("RMSE = " + str(rmse))
        predm1 = pd.DataFrame(m1.predict([values]))
        predm1.columns = (['price'])
        cl1.write("Output data in the initial scale:")
        cl1.write(predm1)

Predict(m1_dist, [x, cut_to_num[cut], carat])

Predict(m2_dist, [x, y, z, cut_to_num[cut], color_to_num[color], clarity_to_num[clarity], depth, carat, table]) 