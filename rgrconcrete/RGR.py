import streamlit as st
import json
import pickle
import sklearn
import pandas as pd
import tensorflow as tf

st.sidebar.header("Введите значения:")

cement = st.sidebar.slider('Cement:', min_value = 102.0, max_value = 528.0, value = 102.0, step = 0.1)

blast = st.sidebar.slider('Blast_Furnace_Slag:', min_value = 0.0, max_value = 316.1, value = 0.0, step = 0.1)

flyash = st.sidebar.slider('Fly_Ash:', min_value = 0.0, max_value = 200.0, value = 0.0, step = 0.1)

water = st.sidebar.slider('Water:', min_value = 127.0, max_value = 228.0, value = 127.0, step = 0.1)

super = st.sidebar.slider('Superplasticizer:', min_value = 0.0, max_value = 22.1, value = 0.0, step = 0.1)

coarse = st.sidebar.slider('Coarse_Aggregate:', min_value = 801.0, max_value = 1145.0, value = 801.0, step = 0.1)

fine = st.sidebar.slider('Fine_Aggregate:', min_value = 594.0, max_value = 945.0, value = 594.0, step = 0.1)

age = st.sidebar.slider('Age:', min_value = 1, max_value = 120, value = 1, step = 1)

df = pd.DataFrame([[cement, blast, flyash, water, super, coarse, fine, age]])
df.columns = (['Cement','Blast_Furnace_Slag','Fly_Ash','Water','Superplasticizer','Coarse_Aggregate','Fine_Aggregate','Age'])

st.write("Введенные значения:")
st.write(df)

cl1, cl2 = st.columns(2)

m1_file = open(r"C:\Users\Nikita\Desktop\siimorgr\m1.json", "r")
m1_dist = json.load(m1_file)
m1_file.close()

m2_file = open(r"C:\Users\Nikita\Desktop\siimorgr\m2.json", "r")
m2_dist = json.load(m2_file)
m2_file.close()

def Predict(dictt, values):
    mN = dictt["modelName"]
    r2 = dictt["R2"]
    rmse = dictt["RMSE"]

    if (mN == "m2.dump"):
        m2 = tf.keras.models.load_model(r"C:\Users\Nikita\Desktop\siimorgr\m2.keras")

        snx_file = open(r"C:\Users\Nikita\Desktop\siimorgr\scalerNormForX.dump", "rb")
        snx = pickle.load(snx_file)
        snx_file.close()

        sny_file = open(r"C:\Users\Nikita\Desktop\siimorgr\scalerNormForY.dump", "rb")
        sny = pickle.load(sny_file)
        sny_file.close()

        cl2.subheader("Нейронная сеть")
        cl2.write("R2 = " + str(r2))
        cl2.write("RMSE = " + str(rmse))
        cl2.write("Результат в нормализованной форме:")

        predm2 = pd.DataFrame(m2.predict(snx.transform([values])))
        predm2.columns = (['Concrete_compressive_strength'])
        cl2.write(predm2)
        cl2.write("Результат в привычной форме:")
        predm2init = pd.DataFrame(sny.inverse_transform([[predm2.iloc[0, 0]]]))
        predm2init.columns = (['Concrete_compressive_strength'])
        cl2.write(predm2init)
    else:
        m1_out = open(r"C:\Users\Nikita\Desktop\siimorgr\m1.dump", "rb")
        m1 = pickle.load(m1_out)
        m1_out.close()

        cl1.subheader("Линейная регрессия")
        cl1.write("R2 = " + str(r2))
        cl1.write("RMSE = " + str(rmse))
        predm1 = pd.DataFrame(m1.predict([values]))
        predm1.columns = (['Concrete_compressive_strength'])
        cl1.write("Результат в привычной форме:")
        cl1.write(predm1)

Predict(m1_dist, [cement, super, age])

Predict(m2_dist, [cement, blast, flyash, water, super, coarse, fine, age]) 