
import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo e o scaler
modelo = joblib.load("modelo_random_forest.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Detetor de Fraudes", layout="centered")

# Título da aplicação
st.title("💳 Detetor de Fraudes com Cartão de Crédito")
st.write("Esta aplicação utiliza um modelo de Machine Learning (Random Forest) para prever se uma transação é fraudulenta ou não.")

# Inputs do utilizador
st.header("📥 Dados da Transação")
amount = st.number_input("Valor da Transação (€)", min_value=0.0, format="%.2f")

# Botão para previsão
if st.button("🔍 Verificar Fraude"):
    nova_transacao = pd.DataFrame({
        'Time': [0],
        'V1': [0], 'V2': [0], 'V3': [0], 'V4': [0], 'V5': [0],
        'V6': [0], 'V7': [0], 'V8': [0], 'V9': [0], 'V10': [0],
        'V11': [0], 'V12': [0], 'V13': [0], 'V14': [0], 'V15': [0],
        'V16': [0], 'V17': [0], 'V18': [0], 'V19': [0], 'V20': [0],
        'V21': [0], 'V22': [0], 'V23': [0], 'V24': [0], 'V25': [0],
        'V26': [0], 'V27': [0], 'V28': [0], 'Amount': [amount]
    })

    nova_transacao['Amount'] = scaler.transform(nova_transacao[['Amount']])

    previsao = modelo.predict(nova_transacao)[0]
    prob = modelo.predict_proba(nova_transacao)[0][1]

    if previsao == 1:
        st.error(f"⚠️ Esta transação é **fraudulenta** com probabilidade de {prob:.2%}")
    else:
        st.success(f"✅ Esta transação **não é fraudulenta** (probabilidade de fraude: {prob:.2%})")
