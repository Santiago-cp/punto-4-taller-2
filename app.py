import streamlit as st
from pydantic import BaseModel
import pandas as pd
import pickle
import tempfile  # Biblioteca para crear archivos temporales
import shutil 
from typing import Optional
from typing import ClassVar
from sklearn.linear_model import Ridge
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

# Cargar el modelo preentrenado desde el archivo pickle
#model_path = "best_model.pkl"
with open("modelo_ridge.pkl", 'rb') as model_file:
    dt2 = pickle.load(model_file)

#prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

# Título de la API
st.title("API de Predicción Académica")

# Botón para subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

# Botón para predecir
if st.button("Predecir"):
    if uploaded_file is not None:
        try:
            # Cargar el archivo subido
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            if uploaded_file.name.endswith(".csv"):
                prueba = pd.read_csv(tmp_path,header = 0,sep=";",decimal=",")
            else:
                prueba = pd.read_excel(tmp_path)

            df_test = prueba.copy()
            predictions = predict_model(dt2, data=df_test)
            predictions["price"] = predictions["prediction_label"]

            # Preparar archivo para descargar
            kaggle = pd.DataFrame({'Email': prueba["Email"], 'price': predictions["price"]})

            # Mostrar predicciones en pantalla
            st.write("Predicciones generadas correctamente!")
            st.write(kaggle)

            # Botón para descargar el archivo de predicciones
            st.download_button(label="Descargar archivo de predicciones",
                               data=kaggle.to_csv(index=False),
                               file_name="kaggle_predictions.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Por favor, cargue un archivo válido.")

# Botón para reiniciar la página
if st.button("Reiniciar"):
    st.experimental_rerun()





