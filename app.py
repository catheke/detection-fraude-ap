import streamlit as st
import numpy as np
import joblib
import os

# Caminhos dos ficheiros
MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "scaler.pkl"


# Carregar modelo e scaler
rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Configurações da página
st.set_page_config(page_title="🛡️ Detector de Fraudes", layout="wide")

st.title("🛡️ Sistema de Detecção de Fraudes com Cartões de Crédito")
st.markdown("""
Este sistema permite prever se uma transação é **fraudulenta ou legítima**, com base num modelo treinado com **Random Forest** sobre um conjunto real de transações.
""")

# Aba principal
aba = st.tabs(["🔍 Verificação de Transação", "📘 Sobre a Aplicação", "👨🏾‍🎓 Sobre Mim"])

# --- Aba 1: Verificação ---
with aba[0]:
    st.subheader("🔎 Preenche os campos abaixo para verificar a transação:")

    with st.expander("ℹ️ O que significam estes campos?"):
        st.markdown("""
        - Os campos `V1` a `V28` são variáveis numéricas resultantes de uma **redução de dimensionalidade (PCA)** feita no conjunto original.
        - `Normalized Time`: segundos desde a primeira transação, normalizados.
        - `Normalized Amount`: valor da transação, também normalizado.
        - Este sistema foi treinado com dados reais e deteta **anormalidades estatísticas** que indicam possíveis fraudes.
        """)

    col1, col2, col3 = st.columns(3)
    v_inputs = []
    for i in range(1, 29):
        with [col1, col2, col3][(i - 1) % 3]:
            val = st.number_input(f"V{i}", value=0.0, format="%.6f")
            v_inputs.append(val)

    with col1:
        normalized_time = st.number_input("Normalized Time", value=0.0, format="%.6f")
    with col2:
        normalized_amount = st.number_input("Normalized Amount", value=0.0, format="%.6f")

    input_data = np.array([*v_inputs, normalized_time, normalized_amount]).reshape(1, -1)

    st.markdown("---")
    if st.button("🔍 Verificar Fraude"):
        prediction = rf_model.predict(input_data)[0]
        prediction_proba = rf_model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 Alerta! Esta transação é suspeita de **fraude**.\n\nProbabilidade de fraude: `{prediction_proba:.2%}`")
        else:
            st.success(f"✅ Transação considerada **legítima**.\n\nProbabilidade de fraude: `{prediction_proba:.2%}`")

# --- Aba 2: Sobre a Aplicação ---
with aba[1]:
    st.subheader("📘 Informações Técnicas")
    st.markdown("""
    - 🔍 **Objetivo**: Prever se uma transação é fraudulenta com base em dados anonimizados de cartões de crédito.
    - ⚙️ **Técnicas usadas**:
        - PCA (redução de dimensionalidade)
        - SMOTE (balanceamento de dados)
        - Random Forest (classificador)
        - Scikit-learn + Streamlit
    - 📊 **Dados**: Contêm 284.807 transações, com apenas 492 fraudes reais.
    - 💾 Os ficheiros `rf_model.pkl` (modelo) e `scaler.pkl` (normalizador) são carregados automaticamente.
    """)

# --- Aba 3: Sobre Mim ---
with aba[2]:
    st.subheader("👨🏾‍🎓 Sobre o Autor")
    st.markdown("""
    Olá! Sou **Pedro Calenga**, estudante do **3.º ano de Ciência da Computação** na  
    **Universidade Mandume ya Ndemufayo – Instituto Politécnico da Huíla** 🇦🇴.

    Este projeto foi desenvolvido como parte dos meus estudos para aplicar **inteligência artificial**  
    no combate a fraudes financeiras. Obrigado por testar a aplicação! 😊

    [💬 Contacta-me no LinkedIn](https://www.linkedin.com) | [📧 Email](mailto:mended2003@gmail.com)
    """)
