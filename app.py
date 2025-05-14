import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from datetime import datetime
import random

# Configuração da página com tema personalizado
st.set_page_config(page_title="Detetor de Fraudes Avançado", layout="wide", initial_sidebar_state="expanded")

# Carregar o modelo e o scaler
try:
    modelo = joblib.load("modelo_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Erro: Arquivos 'modelo_random_forest.pkl' ou 'scaler.pkl' não encontrados. Verifique o diretório.")
    st.stop()

# Estilização personalizada
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; }
    .stNumberInput { background-color: #ffffff; border: 1px solid #ced4da; border-radius: 5px; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    .stMarkdown h1, h2, h3 { color: #2c3e50; }
    .fraud-alert { background-color: #ff4d4f; color: white; padding: 10px; border-radius: 5px; }
    .safe-alert { background-color: #28a745; color: white; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Título e introdução
st.title("💳 Detetor de Fraudes com Cartão de Crédito")
st.markdown("""
Bem-vindo ao **Detetor de Fraudes Avançado**, uma aplicação inovadora desenvolvida por **Pedro Calenga**, estudante da **Universidade Mandume - Instituto Politécnico da Huíla**. 
Utilizando um modelo de Machine Learning **Random Forest**, esta ferramenta detecta transações fraudulentas com precisão, oferecendo uma interface interativa e visualizações avançadas para bancos, fintechs e utilizadores finais. Explore as abas abaixo para verificar transações, analisar o modelo ou conhecer mais sobre o projeto!
""")

# Função para coletar dados da transação
def get_transaction_data():
    st.header("📥 Inserir Dados da Transação")
    with st.form("transaction_form"):
        st.markdown("Preencha os dados para verificar se a transação é fraudulenta:")
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Valor da Transação (€)", min_value=0.0, value=0.0, format="%.2f", help="Valor da transação em euros.")
            time = st.number_input("Tempo (segundos)", min_value=0, value=0, help="Tempo desde a primeira transação registrada.")
        with col2:
            v1 = st.number_input("V1 (Feature PCA)", value=0.0, format="%.6f", help="Componente PCA V1.")
            v2 = st.number_input("V2 (Feature PCA)", value=0.0, format="%.6f", help="Componente PCA V2.")
        with col3:
            v3 = st.number_input("V3 (Feature PCA)", value=0.0, format="%.6f", help="Componente PCA V3.")
            v4 = st.number_input("V4 (Feature PCA)", value=0.0, format="%.6f", help="Componente PCA V4.")
        submit = st.form_submit_button("🔍 Verificar Fraude")
    return amount, time, v1, v2, v3, v4, submit

# Função para plotar importância das features (Plotly)
def plot_feature_importance():
    feature_importance = modelo.feature_importances_
    features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
    
    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 title="Top 10 Features Mais Importantes",
                 color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(xaxis_title="Importância Relativa", yaxis_title="Feature", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Função para plotar matriz de confusão (Plotly)
def plot_confusion_matrix(y_true=None, y_pred=None):
    if y_true is None or y_pred is None:
        y_true = [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # Dados simulados
        y_pred = [0, 1, 0, 1, 1, 0, 0, 0, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predito", y="Real", color="Contagem"),
                    x=["Não Fraude", "Fraude"], y=["Não Fraude", "Fraude"])
    fig.update_layout(title="Matriz de Confusão", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Função para plotar curva ROC (Plotly)
def plot_roc_curve(y_true=None, y_scores=None):
    if y_true is None or y_scores is None:
        y_true = [0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        y_scores = [0.1, 0.9, 0.2, 0.3, 0.8, 0.7, 0.1, 0.2, 0.9, 0.3]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Curva ROC (AUC = {roc_auc:.2f})", line=dict(color="darkorange")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório", line=dict(color="navy", dash="dash")))
    fig.update_layout(title="Curva ROC", xaxis_title="Taxa de Falsos Positivos", yaxis_title="Taxa de Verdadeiros Positivos",
                      height=400, xaxis_range=[0, 1], yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

# Função para criar gauge de probabilidade
def plot_fraud_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Probabilidade de Fraude (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ff4d4f" if prob > 0.5 else "#28a745"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Função para exibir histórico simulado de transações
def show_transaction_history():
    st.subheader("📜 Histórico de Transações (Simulado)")
    history_data = pd.DataFrame({
        "Data": [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(5)],
        "Valor (€)": [random.uniform(10, 1000) for _ in range(5)],
        "Prob. Fraude (%)": [random.uniform(0, 100) for _ in range(5)],
        "Status": [random.choice(["Fraude", "Não Fraude"]) for _ in range(5)]
    })
    st.dataframe(history_data, use_container_width=True)

# Função para exibir explicação técnica
def show_technical_explanation():
    st.markdown("""
    ### Por que Random Forest?
    O **Random Forest** foi escolhido por ser ideal para detecção de fraudes:
    - **Robustez**: Combina várias árvores de decisão, reduzindo overfitting e lidando com dados desbalanceados (poucas fraudes vs. muitas transações legítimas).
    - **Alta Sensibilidade**: Maximiza o **recall**, essencial para minimizar **falsos negativos** (fraudes não detectadas), que causam perdas financeiras.
    - **Interpretabilidade**: A importância das features ajuda a entender quais variáveis (ex.: V1, Amount) influenciam as previsões.
    - **Eficiência**: Rápido para treinar e prever, mesmo em datasets grandes, tornando-o prático para aplicações em tempo real.

    ### Falsos Positivos e Negativos
    - **Falsos Positivos (FP)**: Transações legítimas classificadas como fraudulentas. Podem incomodar clientes (ex.: bloqueio indevido do cartão). O modelo busca minimizá-los para melhorar a experiência do usuário.
    - **Falsos Negativos (FN)**: Fraudes não detectadas, que são mais críticas devido às perdas financeiras. O modelo prioriza alto recall para reduzir FN.
    - **Matriz de Confusão**: Mostra como o modelo equilibra acertos e erros.
    - **Curva ROC**: O AUC (Área sob a Curva) próximo de 1 indica excelente capacidade de distinguir fraudes de não fraudes.

    ### Conceitos de Machine Learning
    - **Pré-processamento**: As features V1-V28 são anonimizadas via PCA para proteger dados sensíveis. O `Amount` é normalizado com `StandardScaler` para consistência.
    - **Treinamento**: O modelo foi treinado em um dataset de transações (ex.: Kaggle Credit Card Fraud), usando técnicas como pesos de classe para lidar com desbalanceamento.
    - **Avaliação**: Métricas como **Precisão**, **Recall**, **F1-Score** e **AUC-ROC** foram usadas para garantir desempenho robusto.
    - **Desafios**: Dados desbalanceados exigem cuidado, e mudanças nos padrões de fraude podem requerer retraining.

    ### Aplicações e Impacto
    - **Casos de Uso**: Monitoramento em tempo real, validação manual de transações suspeitas, integração em sistemas de pagamento.
    - **Impacto**: Reduz perdas financeiras, aumenta a confiança dos clientes e agiliza a detecção de fraudes.
    - **Escalabilidade**: Pode ser integrado via API ou hospedado em cloud para processar grandes volumes de dados.
    """)
    st.subheader("📊 Visualizações do Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matriz de Confusão (Simulada)**")
        plot_confusion_matrix()
    with col2:
        st.markdown("**Curva ROC (Simulada)**")
        plot_roc_curve()
    st.markdown("**Importância das Features**")
    plot_feature_importance()

# Função para exibir introdução do desenvolvedor
def show_developer_intro():
    with st.expander("👨‍💻 Sobre o Desenvolvedor - Pedro Calenga"):
        st.markdown("""
        ### Pedro Calenga - Estudante e Desenvolvedor
        Sou **Pedro Calenga**, estudante da **Universidade Mandume - Instituto Politécnico da Huíla**, apaixonado por tecnologia, inteligência artificial e soluções que impactam o mundo real. Este projeto reflete meu compromisso com a inovação e a excelência técnica.

        - **Formação**: Estudante de Engenharia Informática, com foco em Machine Learning e desenvolvimento de software.
        - **Habilidades**:
          - **Programação**: Python, Scikit-learn, Pandas, Streamlit, Plotly.
          - **Machine Learning**: Desenvolvimento e avaliação de modelos preditivos, com ênfase em problemas do mundo real como detecção de fraudes.
          - **Design**: Criação de interfaces intuitivas e visualizações interativas para comunicar resultados a públicos diversos.
        - **Abordagem**: Priorizo código limpo, modular e documentado, com foco na usabilidade e escalabilidade. Este projeto combina ciência de dados e desenvolvimento web para criar uma ferramenta prática e visualmente atraente.
        - **Visão Futura**: Planejo expandir esta aplicação com suporte a múltiplos modelos (ex.: XGBoost, redes neurais), integração com APIs em tempo real e análise de grandes datasets para monitoramento contínuo.
        - **Missão**: Como estudante angolano, quero usar a tecnologia para resolver desafios locais e globais, contribuindo para um ecossistema financeiro mais seguro.

        **Contato**: [pedro.calenga@universidademandume.ao] | [LinkedIn: Pedro Calenga]
        """)

# Função para exibir FAQ
def show_faq():
    with st.expander("❓ Perguntas Frequentes"):
        st.markdown("""
        - **Como o modelo detecta fraudes?** 
          Usa um Random Forest treinado em dados históricos para identificar padrões de transações fraudulentas.
        - **O que são as features V1-V28?**
          São variáveis anonimizadas via PCA, representando características das transações (ex.: comportamento do usuário, localização).
        - **O modelo é 100% preciso?**
          Nenhum modelo é perfeito. Este foi otimizado para minimizar falsos negativos, mas pode haver erros, como mostrado na matriz de confusão.
        - **Posso usar esta ferramenta em produção?**
          Sim, após ajustes como integração com APIs e retraining periódico para novos padrões de fraude.
        - **Por que usar Streamlit?**
          Streamlit permite criar interfaces web interativas rapidamente, ideal para demonstrações e protótipos.
        """)

# Interface principal com abas
st.sidebar.title("Navegação")
st.sidebar.markdown("Desenvolvido por **Pedro Calenga**")
tab1, tab2, tab3 = st.tabs(["🔍 Verificar Transação", "📊 Análise do Modelo", "ℹ️ Sobre o Projeto"])

with tab1:
    # Coleta de dados
    amount, time, v1, v2, v3, v4, submit = get_transaction_data()
    
    if submit:
        with st.spinner("Analisando transação..."):
            # Criar DataFrame com os dados inseridos
            nova_transacao = pd.DataFrame({
                'Time': [time],
                'V1': [v1], 'V2': [v2], 'V3': [v3], 'V4': [v4], 'V5': [0],
                'V6': [0], 'V7': [0], 'V8': [0], 'V9': [0], 'V10': [0],
                'V11': [0], 'V12': [0], 'V13': [0], 'V14': [0], 'V15': [0],
                'V16': [0], 'V17': [0], 'V18': [0], 'V19': [0], 'V20': [0],
                'V21': [0], 'V22': [0], 'V23': [0], 'V24': [0], 'V25': [0],
                'V26': [0], 'V27': [0], 'V28': [0], 'Amount': [amount]
            })

            # Normalizar o Amount
            try:
                nova_transacao['Amount'] = scaler.transform(nova_transacao[['Amount']])
            except Exception as e:
                st.error(f"Erro ao normalizar o valor da transação: {e}")
                st.stop()

            # Fazer previsão
            try:
                previsao = modelo.predict(nova_transacao)[0]
                prob = modelo.predict_proba(nova_transacao)[0][1]
                st.subheader("Resultado da Previsão")
                if previsao == 1:
                    st.markdown(f"<div class='fraud-alert'>⚠️ Esta transação é <b>fraudulenta</b> com probabilidade de {prob:.2%}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='safe-alert'>✅ Esta transação <b>não é fraudulenta</b> (probabilidade de fraude: {prob:.2%})</div>", unsafe_allow_html=True)

                # Gauge de probabilidade
                st.subheader("Indicador de Risco de Fraude")
                plot_fraud_gauge(prob)

                # Histórico simulado
                show_transaction_history()

            except Exception as e:
                st.error(f"Erro ao realizar a previsão: {e}")

with tab2:
    show_technical_explanation()

with tab3:
    show_developer_intro()
    show_faq()

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido por **Pedro Calenga** | Universidade Mandume - Instituto Politécnico da Huíla | 2025")