import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import sklearn

# Configuração do layout da página
st.set_page_config(page_title="Detecção de Fraudes", layout="wide")

# Título principal
st.title("💳 Previsão de Fraudes em Cartões de Crédito")
st.markdown("Desenvolvido por Pedro Calenga")

# Verificar e carregar o modelo e o scaler
model_path = 'rf_model.pkl'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error("Erro: Arquivos 'rf_model.pkl' ou 'scaler.pkl' não encontrados no diretório da aplicação.")
    st.markdown("""
    Por favor, faça download dos arquivos do repositório:
    - [rf_model.pkl](https://github.com/catheke/detection-fraude-ap/blob/main/rf_model.pkl)
    - [scaler.pkl](https://github.com/catheke/detection-fraude-ap/blob/main/scaler.pkl)
    e coloque-os no mesmo diretório que `app_simple.py`.
    """)
    st.stop()

try:
    with open(model_path, 'rb') as model_file:
        rf_model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {str(e)}")
    st.markdown("""
    Este erro pode ocorrer devido a:
    - **Incompatibilidade de versões**: Os arquivos foram gerados numa versão do Python ou scikit-learn diferente da atual.
      - Versão atual do Python: `{}`  
      - Versão atual do scikit-learn: `{}`  
      Regenere os arquivos `.pkl` no Google Colab com as mesmas versões.
    - **Corrupção do arquivo**: Regenere os arquivos no Colab.
    """.format(sys.version, sklearn.__version__))
    st.stop()

# Formulário para entrada de dados
st.subheader("Insira os dados da transação")
cols = st.columns(4)
v_inputs = []
for i in range(1, 29):
    with cols[(i-1) % 4]:
        v_inputs.append(st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}"))

with cols[0]:
    time_input = st.number_input("Time (segundos)", value=0.0, key="time")
with cols[1]:
    amount_input = st.number_input("Amount (valor)", value=0.0, key="amount")

# Botão para previsão
if st.button("Prever"):
    # Preparar os dados de entrada
    input_data = np.array([v_inputs + [time_input, amount_input]])
    input_df = pd.DataFrame(input_data, columns=[f"V{i}" for i in range(1, 29)] + ['Time', 'Amount'])
    
    # Normalizar Time e Amount
    try:
        input_df['Normalized_Amount'] = scaler.transform(input_df[['Amount']])
        input_df['Normalized_Time'] = scaler.transform(input_df[['Time']])
        input_df = input_df.drop(['Time', 'Amount'], axis=1)
    except Exception as e:
        st.error(f"Erro ao normalizar os dados: {str(e)}")
        st.stop()
    
    # Fazer previsão
    try:
        prediction = rf_model.predict(input_df)[0]
        prediction_proba = rf_model.predict_proba(input_df)[0]
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {str(e)}")
        st.stop()
    
    # Exibir resultado
    st.subheader("Resultado da Previsão")
    if prediction == 1:
        st.error(f"Transação prevista como FRAUDULENTA (Probabilidade: {prediction_proba[1]:.4f})")
    else:
        st.success(f"Transação prevista como NÃO FRAUDULENTA (Probabilidade: {prediction_proba[0]:.4f})")

# Rodapé
st.markdown("""
---
**Desenvolvido por**: Pedro Catheke Mendes Calenga  
**Repositório**: [https://github.com/catheke/detection-fraude-ap](https://github.com/catheke/detection-fraude-ap)
""")
