import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

# Configurar layout
st.set_page_config(page_title="Deteção de Fraudes - Pedro Calenga", layout="centered")

# Estilo CSS claro e vibrante, com caixas explicativas visíveis
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #0066cc;
        text-align: center;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
    }
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }
    .credit-section {
        text-align: center;
        padding: 15px;
        background-color: #0066cc;
        color: white;
        border-radius: 10px;
        margin-top: 20px;
    }
    .explanation-box {
        background-color: #e6f3ff !important;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #b3d4fc;
        margin-bottom: 15px;
        color: #333333 !important;
        font-size: 16px;
    }
    .explanation-box h3 {
        color: #0066cc;
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("🛡️ Deteção de Fraudes em Cartões de Crédito")
st.markdown("**Desenvolvido por Pedro Calenga** para Aprendizagem Computacional")

# Introdução ao Machine Learning
st.markdown("""
<div class='explanation-box'>
<h3>O que é Machine Learning?</h3>
<p>Machine Learning é como ensinar um computador a ser um detetive: ele analisa exemplos (como transações de cartão) e aprende a encontrar padrões, como fraudes. Neste projeto, usei um modelo chamado <strong>Random Forest</strong>, que combina várias "árvores de decisão" para fazer previsões precisas. Cada árvore analisa os dados de um jeito diferente, e juntas elas decidem se uma transação é normal ou fraude.</p>
<p><strong>Por que Random Forest?</strong> É rápido, lida bem com dados desbalanceados (como nossas 492 fraudes em 284 mil transações), e é mais simples que redes neurais, que precisam de mais dados.</p>
<p><strong>Por que é útil?</strong> Bancos usam isso para proteger clientes, evitando perdas de milhões de euros por ano!</p>
</div>
""", unsafe_allow_html=True)

# Introdução ao Projeto
st.markdown("""
<div class='explanation-box'>
<h3>Sobre o Projeto</h3>
<p>Este projeto usa o dataset <strong>creditcard.csv</strong>, com 284.807 transações reais de cartões de crédito (de um estudo europeu). Apenas 0,17% são fraudes, o que é um desafio, pois o modelo precisa encontrar essas "agulhas no palheiro". Meu objetivo é ajudar bancos a identificar fraudes rapidamente, protegendo os clientes.</p>
<p><strong>Como fiz?</strong> Coletei os dados, usei uma técnica chamada SMOTE para balancear as fraudes no treino, normalizei os dados, treinei um Random Forest, testei sua precisão (95,7% recall) e criei esta aplicação para mostrar os resultados.</p>
<p><strong>Impacto real?</strong> Bancos e fintechs (como PayPal ou Nubank) usam modelos assim para bloquear fraudes em tempo real, salvando milhões!</p>
</div>
""", unsafe_allow_html=True)

# Explicação inicial
st.write("""
Esta aplicação usa inteligência artificial para encontrar fraudes em transações de cartão de crédito. 
Carregue o arquivo de dados, e ela mostrará:
- Quantas fraudes foram encontradas.
- 50 exemplos de transações (normais, fraudes, erros).
- Gráficos interativos e estatísticas detalhadas.
""")

# Carregar modelo e scaler
try:
    modelo = joblib.load("models/modelo_fraude.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error("Erro: Modelos não encontrados. Verifique a pasta 'models'.")
    st.stop()

# Upload do CSV
st.subheader("1. Carregar os Dados")
st.markdown("""
<div class='explanation-box'>
<h3>Por que carregar um arquivo?</h3>
<p>O arquivo <strong>creditcard.csv</strong> contém 284.807 transações, cada uma com colunas como <strong>Amount</strong> (valor em euros), <strong>Time</strong> (quando aconteceu) e <strong>V1 a V28</strong> (características secretas, como padrões de gasto, que ajudam o modelo a detectar fraudes). A coluna <strong>Class</strong> diz se é fraude (1) ou normal (0).</p>
<p><strong>Pré-processamento</strong>: Normalizei os dados (scaler) para comparar valores justamente e usei SMOTE para dar mais exemplos de fraudes ao modelo durante o treino, como ensinar o detetive a reconhecer pistas raras.</p>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Escolha o arquivo CSV (ex.: creditcard.csv)", type=["csv"])

if uploaded_file is not None:
    # Ler CSV
    try:
        dados = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {str(e)}")
        st.stop()

    # Verificar colunas
    colunas_necessarias = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                          'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                          'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                          'V28', 'Amount']
    if all(col in dados.columns for col in colunas_necessarias):
        # Proporção de classes
        if 'Class' in dados.columns:
            st.subheader("2. Proporção de Classes")
            proporcao = dados['Class'].value_counts(normalize=True)
            st.metric("Normais (0)", f"{proporcao[0]:.4%} ({int(proporcao[0] * len(dados))})")
            st.metric("Fraudes (1)", f"{proporcao[1]:.4%} ({int(proporcao[1] * len(dados))})", delta="🕵️")
            st.markdown("""
            <div class='explanation-box'>
            <h3>O que significa proporção?</h3>
            <p>Das 284.807 transações, apenas 0,17% (cerca de 492) são fraudes. Isso é como encontrar 492 agulhas num palheiro de 284 mil! O modelo precisa ser muito preciso para não confundir normais com fraudes.</p>
            </div>
            """, unsafe_allow_html=True)

        # Normalizar e prever
        dados_input = dados[colunas_necessarias]
        try:
            dados_scaled = scaler.transform(dados_input)
        except Exception as e:
            st.error(f"Erro na normalização: {str(e)}")
            st.stop()

        previsoes = modelo.predict(dados_scaled)
        dados['Resultado'] = ['🔴 Fraude' if x == 1 else '🟢 Normal' for x in previsoes]

        # Estatísticas
        st.subheader("3. Resultados Gerais")
        total_fraudes = sum(previsoes)
        valor_medio_normal = dados[dados['Resultado'] == '🟢 Normal']['Amount'].mean()
        valor_medio_fraude = dados[dados['Resultado'] == '🔴 Fraude']['Amount'].mean() if total_fraudes > 0 else 0
        valor_max_normal = dados[dados['Resultado'] == '🟢 Normal']['Amount'].max()
        valor_max_fraude = dados[dados['Resultado'] == '🔴 Fraude']['Amount'].max() if total_fraudes > 0 else 0
        valor_std_normal = dados[dados['Resultado'] == '🟢 Normal']['Amount'].std()
        valor_std_fraude = dados[dados['Resultado'] == '🔴 Fraude']['Amount'].std() if total_fraudes > 0 else 0
        st.metric("Fraudes Detectadas", total_fraudes, delta="🕵️")
        st.write(f"**Valor médio (normais)**: €{valor_medio_normal:.2f} (máximo: €{valor_max_normal:.2f}, desvio: €{valor_std_normal:.2f})")
        st.write(f"**Valor médio (fraudes)**: €{valor_medio_fraude:.2f} (máximo: €{valor_max_fraude:.2f}, desvio: €{valor_std_fraude:.2f})")
        st.markdown("""
        <div class='explanation-box'>
        <h3>Por que essas estatísticas?</h3>
        <p>A média mostra se fraudes têm valores diferentes (ex.: fraudes podem ser mais altas). O máximo indica o maior gasto, e o desvio padrão mostra o quanto os valores variam. Isso ajuda o modelo a entender o comportamento dos fraudadores!</p>
        </div>
        """, unsafe_allow_html=True)

        # Matriz de confusão
        if 'Class' in dados.columns:
            st.subheader("4. Desempenho do Modelo")
            cm = confusion_matrix(dados['Class'], previsoes)
            cm_df = pd.DataFrame(cm, index=['Normal (0)', 'Fraude (1)'], columns=['Previsto Normal', 'Previsto Fraude'])
            st.write("**Matriz de Confusão (Tabela)**:")
            st.write(cm_df)
            # Heatmap da matriz
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Previsto Normal', 'Previsto Fraude'], 
                        yticklabels=['Normal (0)', 'Fraude (1)'])
            ax.set_title('Matriz de Confusão')
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("""
            <div class='explanation-box'>
            <h3>O que é a matriz de confusão?</h3>
            <p>É como o boletim do detetive, mostrando acertos e erros:</p>
            <ul>
                <li><strong>Verdadeiros Positivos (TP)</strong>: Fraudes acertadas (ex.: ~471).</li>
                <li><strong>Falsos Positivos (FP)</strong>: Normais marcadas como fraudes (ex.: 6).</li>
                <li><strong>Verdadeiros Negativos (TN)</strong>: Normais corretas (ex.: 284.000+).</li>
                <li><strong>Falsos Negativos (FN)</strong>: Fraudes perdidas (ex.: ~21).</li>
            </ul>
            <p><strong>Por que isso importa?</strong> Falsos positivos incomodam clientes (cartão bloqueado sem motivo), e falsos negativos deixam fraudes passarem. Queremos poucos dos dois!</p>
            </div>
            """, unsafe_allow_html=True)
            falsos_positivos = cm[0, 1]
            falsos_negativos = cm[1, 0]
            st.metric("Falsos Positivos", falsos_positivos, delta="⚠️")
            st.metric("Falsos Negativos", falsos_negativos, delta="⚠️")

            # Relatório de classificação
            report = classification_report(dados['Class'], previsoes, digits=4, output_dict=True)
            st.write("**Métricas de Desempenho**:")
            st.write(pd.DataFrame(report).transpose())
            st.markdown("""
            <div class='explanation-box'>
            <h3>O que são essas métricas?</h3>
            <p>São como notas do detetive:</p>
            <ul>
                <li><strong>Precisão</strong>: De todas as transações marcadas como fraude, quantas eram realmente fraudes? (Ex.: ~98%).</li>
                <li><strong>Recall</strong>: De todas as fraudes reais, quantas pegamos? (Ex.: 95,7%).</li>
                <li><strong>F1-score</strong>: Equilíbrio entre precisão e recall, como uma nota final.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # Tabela de transações
        st.subheader("5. Exemplos de Transações")
        st.write("Aqui estão 50 transações, incluindo normais, fraudes, falsos positivos e negativos:")
        colunas_exibir = ['Amount', 'Time', 'Resultado']
        if 'Class' in dados.columns:
            colunas_exibir.append('Class')
        normais = dados[dados['Resultado'] == '🟢 Normal'][colunas_exibir].head(20)
        fraudes = dados[dados['Resultado'] == '🔴 Fraude'][colunas_exibir].head(20)
        if 'Class' in dados.columns:
            falsos_positivos_df = dados[(dados['Class'] == 0) & (dados['Resultado'] == '🔴 Fraude')][colunas_exibir].head(5)
            falsos_negativos_df = dados[(dados['Class'] == 1) & (dados['Resultado'] == '🟢 Normal')][colunas_exibir].head(5)
            amostra = pd.concat([normais, fraudes, falsos_positivos_df, falsos_negativos_df])
            st.markdown("""
            <div class='explanation-box'>
            <h3>Legenda da Tabela</h3>
            <p>Esta tabela mostra 50 transações para entender o que o modelo fez:</p>
            <ul>
                <li><strong>Amount</strong>: Valor da transação em euros (ex.: €529).</li>
                <li><strong>Time</strong>: Quando a transação aconteceu (em segundos).</li>
                <li><strong>Resultado</strong>: 🟢 Normal (segura) ou 🔴 Fraude (suspeita).</li>
                <li><strong>Class</strong>: Verdade real (0 = normal, 1 = fraude).</li>
                <li><strong>Falsos Positivos</strong>: Normais que o modelo achou que eram fraudes.</li>
                <li><strong>Falsos Negativos</strong>: Fraudes que o modelo achou que eram normais.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            amostra = pd.concat([normais, fraudes])
            st.markdown("""
            <div class='explanation-box'>
            <h3>Legenda da Tabela</h3>
            <p>Esta tabela mostra 40 transações (20 normais, 20 fraudes):</p>
            <ul>
                <li><strong>Amount</strong>: Valor da transação em euros.</li>
                <li><strong>Time</strong>: Quando a transação aconteceu (em segundos).</li>
                <li><strong>Resultado</strong>: 🟢 Normal (segura) ou 🔴 Fraude (suspeita).</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.dataframe(amostra, use_container_width=True)
        # Exportação de CSV
        csv = amostra.to_csv(index=False)
        st.download_button(
            label="Baixar Tabela (CSV)",
            data=csv,
            file_name="transacoes_resultados.csv",
            mime="text/csv"
        )

        # Gráfico interativo Plotly
        st.subheader("6. Gráfico Interativo de Transações")
        st.write("Explore as transações por valor e tempo, com cores para normais e fraudes:")
        fig_plotly = px.scatter(
            dados.head(1000),  # Limitar para performance
            x='Time', y='Amount',
            color='Resultado',
            color_discrete_map={'🟢 Normal': '#00cc00', '🔴 Fraude': '#ff3333'},
            title="Transações: Tempo vs. Valor",
            labels={'Time': 'Tempo (segundos)', 'Amount': 'Valor (€)'},
            hover_data=['Resultado']
        )
        st.plotly_chart(fig_plotly, use_container_width=True)
        st.markdown("""
        <div class='explanation-box'>
        <h3>Por que este gráfico?</h3>
        <p>Este gráfico interativo mostra como normais (🟢) e fraudes (🔴) aparecem em diferentes tempos e valores. Passe o mouse para ver detalhes! Fraudes podem formar padrões, como valores altos em certos momentos.</p>
        </div>
        """, unsafe_allow_html=True)

        # Mapa de calor
        if total_fraudes > 0:
            st.subheader("7. Correlação entre Características")
            st.write("Este mapa mostra como as características (V1 a V28) se relacionam nas fraudes:")
            fraudes = dados[dados['Resultado'] == '🔴 Fraude'][colunas_necessarias]
            corr = fraudes.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap='coolwarm', ax=ax, center=0)
            ax.set_title('Correlações nas Transações Fraudulentas')
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("""
            <div class='explanation-box'>
            <h3>O que é correlação?</h3>
            <p>Correlação mostra se duas características (ex.: V1 e V2) mudam juntas. Cores fortes (vermelho ou azul escuro) indicam que essas características são pistas importantes para o detetive. Por exemplo, se V1 aumenta quando V2 aumenta, o modelo usa isso para encontrar fraudes.</p>
            </div>
            """, unsafe_allow_html=True)

        # FAQ
        st.subheader("8. Perguntas Frequentes")
        st.markdown("""
        <div class='explanation-box'>
        <h3>Perguntas que você pode ter</h3>
        <p><strong>Por que o modelo erra?</strong> Fraudes são raras (0,17%), então o modelo às vezes é cauteloso demais (falsos positivos) ou perde algumas fraudes (falsos negativos).</p>
        <p><strong>Como bancos usam isso?</strong> Eles analisam transações marcadas como fraudes e bloqueiam cartões se necessário, protegendo clientes.</p>
        <p><strong>Por que V1 a V28 são secretas?</strong> São dados transformados para proteger a privacidade, mas representam padrões de gasto, como horário ou localização.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        missing_cols = [col for col in colunas_necessarias if col not in dados.columns]
        st.error(f"O CSV deve conter todas as colunas necessárias. Faltam: {', '.join(missing_cols)}.")

# Seção de créditos
st.markdown("---")
st.markdown("""
    <div class='credit-section'>
        <h2>Créditos</h2>
        <p><strong>Desenvolvido por</strong>: Pedro Calenga</p>
        <p><strong>Instituição</strong>: [Insira o nome da tua instituição]</p>
        <p><strong>Disciplina</strong>: Aprendizagem Computacional</p>
        <p><strong>Ano</strong>: 2025</p>
        <p>Esta aplicação usa Machine Learning para detectar fraudes em cartões de crédito, com explicações simples e resultados precisos para proteger bancos e clientes!</p>
    </div>
""", unsafe_allow_html=True)