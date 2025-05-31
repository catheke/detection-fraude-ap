import streamlit as st
from pages import treino_modelo, previsao_fraude

st.set_page_config(page_title="Deteção de Fraudes em Cartões de Crédito", layout="wide")

st.title("💳 Deteção de Fraudes em Cartões de Crédito")

st.write("Escolhe uma opção no menu à esquerda.")
