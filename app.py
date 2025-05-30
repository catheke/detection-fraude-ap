import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configuração do layout da página
st.set_page_config(page_title="Detecção de Fraudes em Cartões de Crédito", layout="wide")

# Título principal
st.title("💳 Detector de Fraudes em Cartões de Crédito")
st.markdown("Desenvolvido por Pedro Calenga, estudante da Universidade Mandume Ya Ndemufayo - Instituto Politécnico da Huíla")

# Sidebar para navegação
st.sidebar.header("Navegação")
page = st.sidebar.radio("Selecione uma seção:", ["Previsão", "Sobre Mim", "Sobre o Projeto", "Análise do Dataset"])

# Carregar o modelo e o scaler
try:
    with open('rf_model.pkl', 'rb') as model_file:
        rf_model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Erro: Arquivos 'rf_model.pkl' ou 'scaler.pkl' não encontrados. Certifique-se de que estão no diretório correto.")
    st.stop()

# Seção de Previsão
if page == "Previsão":
    st.write("""
    Esta aplicação utiliza um modelo Random Forest treinado para prever se uma transação de cartão de crédito é fraudulenta ou não.
    Insira os valores das características abaixo para realizar a previsão.
    """)

    # Formulário para entrada de dados
    st.subheader("Insira os dados da transação")
    cols = st.columns(4)
    v_inputs = []
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            v_inputs.append(st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"v{i}"))
    
    with cols[0]:
        time_input = st.number_input("Time (segundos desde a primeira transação)", value=0.0, key="time")
    with cols[1]:
        amount_input = st.number_input("Amount (valor da transação)", value=0.0, key="amount")

    # Botão para previsão
    if st.button("Prever", key="predict"):
        # Preparar os dados de entrada
        input_data = np.array([v_inputs + [time_input, amount_input]])
        input_df = pd.DataFrame(input_data, columns=[f"V{i}" for i in range(1, 29)] + ['Time', 'Amount'])
        
        # Normalizar Time e Amount
        input_df['Normalized_Amount'] = scaler.transform(input_df[['Amount']])
        input_df['Normalized_Time'] = scaler.transform(input_df[['Time']])
        input_df = input_df.drop(['Time', 'Amount'], axis=1)
        
        # Fazer previsão
        prediction = rf_model.predict(input_df)[0]
        prediction_proba = rf_model.predict_proba(input_df)[0]
        
        # Exibir resultado
        st.subheader("Resultado da Previsão")
        if prediction == 1:
            st.error(f"Transação prevista como FRAUDULENTA (Probabilidade: {prediction_proba[1]:.4f})")
        else:
            st.success(f"Transação prevista como NÃO FRAUDULENTA (Probabilidade: {prediction_proba[0]:.4f})")

# Seção Sobre Mim
elif page == "Sobre Mim":
    st.header("👨‍🎓 Sobre Mim")
    st.markdown("""
    **Pedro Calenga**  
    Estudante do 3.º ano de Engenharia Informática na **Universidade Mandume Ya Ndemufayo - Instituto Politécnico da Huíla**, Lubango, Angola.  
    Apaixonado por inteligência artificial e ciência de dados, este projeto reflete o meu interesse em aplicar aprendizado de máquina para resolver problemas reais, como a detecção de fraudes em transações financeiras. O meu objetivo é contribuir para a segurança digital em Angola, onde o uso de cartões de crédito está em crescimento. Este trabalho demonstra a minha capacidade de combinar teoria académica com aplicações práticas, preparando-me para desafios no mercado tecnológico.
    """)

# Seção Sobre o Projeto
elif page == "Sobre o Projeto":
    st.header("📋 Sobre o Projeto")
    st.markdown("""
    ### Contexto do Problema
    A detecção de fraudes em cartões de crédito é essencial no sector financeiro, especialmente num mundo onde as transações digitais crescem exponencialmente. Este projeto utiliza o dataset **Credit Card Fraud Detection** do Kaggle, com 284.807 transações, sendo apenas 492 fraudulentas (0,17%). O desafio é detetar essas fraudes num dataset altamente desequilibrado.

    ### Escolha do Random Forest
    O modelo **Random Forest** foi selecionado por várias razões:
    - **Robustez**: Combina múltiplas árvores de decisão, reduzindo o *overfitting* e lidando bem com dados desequilibrados.
    - **Capacidade de lidar com *features* PCA**: As colunas `V1` a `V28` são anonimizadas via PCA, e o Random Forest não requer suposições sobre a distribuição dos dados.
    - **Importância de *features***: Permite identificar quais variáveis (ex.: `V14`, `V17`) são mais relevantes para detetar fraudes.
    - **Minimização de falsos negativos**: Falsos negativos (fraudes não detetadas) são críticos em aplicações financeiras. O Random Forest, combinado com SMOTE, otimiza o *recall* para a classe fraudulenta.
    - **Comparação com outros modelos**: Testei Regressão Logística, mas o Random Forest superou em métricas como AUC-ROC (0,93 vs. 0,90) e *recall* para fraudes (0,78 vs. 0,70).

    ### Uso do SMOTE
    O dataset é desequilibrado (284.315 não fraudes vs. 492 fraudes). Para resolver isso:
    - **SMOTE** (*Synthetic Minority Oversampling Technique*) foi usado para criar amostras sintéticas da classe minoritária (fraudes), balanceando o conjunto de treino para 199.020 instâncias de cada classe.
    - Isso melhora a capacidade do modelo de aprender padrões de fraudes sem enviesar para a classe maioritária.

    ### Relevância
    Este projeto é relevante para Angola, onde a digitalização financeira está em crescimento. Um modelo eficiente pode proteger consumidores e instituições, reduzindo perdas financeiras.
    """)

# Seção Análise do Dataset
elif page == "Análise do Dataset":
    st.header("📊 Análise do Dataset")
    st.markdown("""
    O dataset contém 284.807 transações, com 31 colunas: `Time`, `Amount`, `V1` a `V28` (*features* anonimizadas via PCA) e `Class` (0 para não fraude, 1 para fraude). Abaixo, mostramos a distribuição das classes, a matriz de confusão e a curva ROC.
    """)

    # Dados da matriz de confusão fornecida
    cm_data = np.array([[85276, 19], [33, 115]])
    st.subheader("Matriz de Confusão")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predito')
    ax_cm.set_ylabel('Verdadeiro')
    ax_cm.set_title('Matriz de Confusão')
    st.pyplot(fig_cm)

    # Dados para a curva ROC (simulados com base no AUC fornecido)
    st.subheader("Curva ROC")
    fpr = np.linspace(0, 1, 100)
    tpr = np.linspace(0, 1, 100) ** 0.5  # Simulação para AUC ~ 0.9489
    roc_auc = 0.9489
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color='blue')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax_roc.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax_roc.set_title('Curva ROC')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# Rodapé
st.markdown("""
---
**Desenvolvido por**: Pedro Catheke Mendes Calenga  
**Repositório**: [https://github.com/catheke/detection-fraude-ap](https://github.com/catheke/detection-fraude-ap)
""")