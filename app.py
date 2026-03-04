import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from openai import OpenAI
import tempfile
import os

import whisper
from streamlit_mic_recorder import mic_recorder
from io import BytesIO
from gtts import gTTS

def preparar_dados_para_narracao(dados, pergunta):
    """
    Reduz os dados para o essencial analítico, sem perder informação relevante.
    """
    if isinstance(dados, pd.DataFrame):
        resumo = {
            "total_linhas": len(dados),
            "colunas": list(dados.columns),
            "estatisticas": dados.describe().round(2).to_dict(),
            "top_5": dados.head(5).to_dict(orient="records"),
            "bottom_5": dados.tail(5).to_dict(orient="records"),
        }
        # Se tiver coluna numérica, adiciona os outliers
        num_cols = dados.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            resumo["maiores_valores"] = dados.nlargest(3, num_cols[0]).to_dict(orient="records")
            resumo["menores_valores"] = dados.nsmallest(3, num_cols[0]).to_dict(orient="records")
        return str(resumo)

    elif isinstance(dados, pd.Series):
        return str({
            "total_itens": len(dados),
            "min": dados.min(),
            "max": dados.max(),
            "media": round(dados.mean(), 2) if dados.dtype != 'object' else "N/A",
            "top_5": dados.head(5).to_dict(),
            "bottom_5": dados.tail(5).to_dict(),
        })

    else:
        # Número, string, resultado simples — manda direto
        return str(dados)
    

def extrair_dados_do_grafico(figura):
    """
    Reconstrói um DataFrame estruturado a partir do gráfico,
    garantindo alinhamento entre categorias e valores.
    """
    try:
        trace = figura.data[0]
        tipo = trace.type

        if tipo == 'pie':
            df_grafico = pd.DataFrame({
                "categoria": list(trace.labels),
                "valor": list(trace.values)
            })

        elif tipo in ['bar', 'scatter', 'line']:
            df_grafico = pd.DataFrame({
                "categoria": list(trace.x),
                "valor": list(trace.y)
            })

        elif tipo == 'histogram':
            df_grafico = pd.DataFrame({"valor": list(trace.x)})

        else:
            return f"Tipo de gráfico '{tipo}' — dados brutos: x={list(trace.x)}, y={list(trace.y)}"

        # Remove nulos e reseta índice — elimina o problema dos índices aparecendo
        df_grafico = df_grafico.dropna().reset_index(drop=True)

        # Serializa como tabela legível, não como dicionário bagunçado
        return df_grafico.to_string(index=False)

    except Exception as e:
        return f"Não foi possível extrair dados do gráfico: {e}"
    
# FUNÇÕES DE SPEECH-TO-TEXT (NOVO)

@st.cache_resource(show_spinner="Carregando modelo de transcrição (Whisper)...")
def carregar_modelo_whisper():
    """Carrega o modelo Whisper apenas uma vez e guarda na memória."""
    return whisper.load_model("base")

def obter_pergunta_usuario():
    """
    Renderiza a interface de entrada de texto e áudio.
    Se houver áudio, transcreve e retorna o texto. Caso contrário, retorna o texto digitado.
    """
    st.write("### 💬 Faça sua Pergunta aos Dados")
    
    pergunta_final = None
    
    col_texto, col_audio = st.columns([4, 1])
    
    with col_texto:
        pergunta_digitada = st.text_input("Digite sua pergunta ou peça um gráfico:", key="input_texto")
        
    with col_audio:
        st.write("Ou fale:")
        # O mic_recorder retorna um dicionário com os bytes do áudio quando a gravação para
        audio_gravado = mic_recorder(start_prompt="🎙️ Gravar", stop_prompt="⏹️ Parar", key='gravador')
        
    if audio_gravado:
        with st.spinner("Transcrevendo seu áudio..."):
            # Salva temporariamente de forma segura para o Whisper ler
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_gravado['bytes'])
                tmp_audio_path = tmp_audio.name
                
            try:
                modelo = carregar_modelo_whisper()
                resultado = modelo.transcribe(tmp_audio_path, language="pt")
                pergunta_final = resultado["text"].strip()
                st.success(f"🗣️ **Você disse:** {pergunta_final}")
            finally:
                # Limpa o arquivo temporário do computador
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)
                    
    elif pergunta_digitada:
        pergunta_final = pergunta_digitada
        
    return pergunta_final

# Configurando o cliente Ollama
client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key= "ollama"
)

st.set_page_config(page_title="Local Data Chat", layout= "wide")
st.title("📊 BI Conversacional (Powered by Llama 3)")

uploaded_file = st.file_uploader("Faça o upload do seu arquivo .CSV", type = ["csv"])

if uploaded_file:
    # 1. Carregamento dos Dados
    df = pd.read_csv(uploaded_file)

    # 2. AutoCleaner (Limpeza silenciosa)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df_temp = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                df[col] = pd.to_numeric(df_temp)
            except ValueError:
                pass 

    # 3. Amostragem Inteligente para EDA (Evitar gargalo de RAM no navegador)
    if df.shape[0] > 50000:
        df_eda = df.sample(50000, random_state=42)
        st.info("⚠️ Dataset muito grande. A seção 'Análise Exploratória' abaixo está usando uma amostra aleatória de 50.000 linhas para garantir performance. O Chat com IA usará os dados completos.")
    else:
        df_eda = df

    # --- INÍCIO DA INTERFACE COM EXPANSORES ---

    # Expansor 1: Pré-visualização
    with st.expander("🔍 Pré-visualização dos Dados", expanded=False):
        total_linhas = df.shape[0]
        st.write(f"**Total de linhas no arquivo original:** {total_linhas}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Primeiras 5 linhas:**")
            st.dataframe(df.head())
        with col2:
            st.write("**Últimas 5 linhas:**")
            st.dataframe(df.tail())

    # Expansor 2: O Auto-EDA (O Analista Júnior Automático)
    with st.expander("📈 Mais informações sobre os dados", expanded=False):
        
        st.write("### 1. Estatísticas Descritivas")
        st.dataframe(df_eda.describe())

        col_nulos, col_corr = st.columns(2)
        
        with col_nulos:
            st.write("### 2. Valores Ausentes (Nulos)")
            nulos = df_eda.isnull().sum()
            nulos = nulos[nulos > 0].sort_values(ascending=False)
            if not nulos.empty:
                st.dataframe(nulos.rename("Quantidade de Nulos"))
            else:
                st.success("Base limpa! Não há valores nulos.")

        with col_corr:
            st.write("### 3. Top 10 Correlações Fortes")
            df_num = df_eda.select_dtypes(include='number')
            
            if df_num.shape[1] >= 2:
                # Calcula a matriz e pega o valor absoluto para ranquear a força
                corr_matrix = df_num.corr()
                
                # Pega só o triângulo superior da matriz para não repetir pares (A-B e B-A)
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Desempilha, remove os nulos e ranqueia pelos maiores valores absolutos
                pares = upper.unstack().dropna()
                pares_abs = pares.abs().sort_values(ascending=False)
                top_10_idx = pares_abs.head(10).index
                
                # Monta a tabela final bonitinha
                top_10_df = pares[top_10_idx].reset_index()
                top_10_df.columns = ['Variável 1', 'Variável 2', 'Correlação']
                st.dataframe(top_10_df)
            else:
                st.info("Não há colunas numéricas suficientes para calcular correlação.")

    # --- FIM DA INTERFACE DE EDA ---
    st.divider()

    # 4. O Chatbot com IA
    df_info = f"Columns: {list(df.columns)}\nData Types: {df.dtypes.to_dict()}"

    pergunta = obter_pergunta_usuario()

    if pergunta:
        with st.spinner("Llama 3 analisando e escrevendo código..."):
            
            # PROMPT MAIS AGRESSIVO
            prompt = f"""You are an expert Python Data Analyst.
            A pandas DataFrame `df` is ALREADY loaded in memory.
            `pandas` is already imported as `pd` and `plotly.express` is already imported as `px`.
            Structure:
            {df_info}

            User query: "{pergunta}"

            Write ONLY the Python code to answer this.
            Rules:
            1. DO NOT use `pd.read_csv()`. The `df` is already loaded.
            2. DO NOT use `import pandas` or `import plotly`. They are already imported.
            3. DO NOT format with markdown (no ```python). Just raw code.
            4. If the user asks for a simple answer (text/number), save it in a variable named `resultado_final`.
            5. IF THE USER ASKS FOR A CHART OR GRAPH (e.g. "gráfico"): You MUST use `px` (plotly.express). DO NOT use `df.plot()` or `matplotlib`. Save the figure exactly in the variable `figura_final`. Do not use fig.show().
            6. If grouping or ranking categories, ALWAYS use `.reset_index()` after `.mean()`, `.sum()` or `.value_counts()` to convert the Series back to a DataFrame before plotting. Example: `df.groupby('col1')['col2'].mean().reset_index()`.
            7. For correlation analysis: ALWAYS call .corr(numeric_only=True) on the FULL DataFrame first, then isolate the target column. NEVER call .corr() on a single-column DataFrame.
            Correct pattern:
            corr_series = df.corr(numeric_only=True)['target_col'].drop('target_col').abs()
            resultado_final = corr_series.sort_values(ascending=False).head(N).reset_index() , where N is the requested number".
            resultado_final.columns = ['Variável', 'Correlação']
            """

            try: 
                response = client.chat.completions.create(
                    model = "llama3",
                    messages = [
                        {"role": "system", "content": "You are a Python code generator. Output raw executable code only."},
                        {"role": "user", "content": str(prompt)}
                    ],
                    temperature=0.0 
                )
                
                codigo_gerado = response.choices[0].message.content
                codigo_limpo = codigo_gerado.replace("```python", "").replace("```", "").strip()

                # --- NOVO GUARDRAIL INTELIGENTE ---
                # Verifica se as variáveis exigidas não estão no código gerado
                if "resultado_final" not in codigo_limpo and "figura_final" not in codigo_limpo:
                    # Se o código usa px (plotly), forçamos a variável de gráfico
                    if "px." in codigo_limpo:
                        codigo_limpo = f"figura_final = {codigo_limpo}"
                    # Caso contrário, forçamos a variável de texto
                    else:
                        codigo_limpo = f"resultado_final = {codigo_limpo}"
                # ---------------------------------

                with st.expander("Ver Código Gerado pela IA"):
                    st.code(codigo_limpo, language="python")

                variaveis_locais = {"df": df, "pd": pd, "px": px} 
                
                try:
                    exec(codigo_limpo, globals(), variaveis_locais)

                    dados_para_explicar = None

                    if "figura_final" in variaveis_locais:
                        st.success("Gráfico gerado com sucesso!")
                        figura = variaveis_locais["figura_final"]
                        st.plotly_chart(figura, use_container_width=True)
                        dados_para_explicar = extrair_dados_do_grafico(figura)
                            
                    elif "resultado_final" in variaveis_locais:
                        st.success("Resposta Encontrada:")
                        resultado = variaveis_locais["resultado_final"]
                        
                        # Se for tabela ou série, usa o st.dataframe
                        if isinstance(resultado, (pd.DataFrame, pd.Series)):
                            st.dataframe(resultado)
                        # Se for um número solto, texto, etc., usa o st.write
                        else:
                            st.write(f"**Resultado:** {resultado}")
                            
                        dados_para_explicar = resultado
                    
                    else:
                        st.warning("Código executado, mas nenhuma variável retornada. Tente reformular a pergunta.")
                    
                    # ETAPA 2 e 3: O NARRADOR E A VOZ
                    if dados_para_explicar is not None:
                        with st.spinner("Gerando explicação em áudio..."):
                            
                            dados_str = preparar_dados_para_narracao(dados_para_explicar, pergunta)
                            
                            prompt_narrador = f"""Você é um Analista de Dados Sênior apresentando resultados em uma reunião executiva.

                            PERGUNTA DO USUÁRIO: "{pergunta}"

                            TABELA DE DADOS EXATOS (use SOMENTE esses valores):
                            {dados_str}

                            INSTRUÇÕES:
                            1. Responda EXCLUSIVAMENTE em texto corrido, em português brasileiro.
                            2. NUNCA escreva código, índices, colchetes ou símbolos de programação.
                            3. Use APENAS os valores da tabela acima. Cada número que citar deve estar na tabela.
                            4. Estruture em: [fato principal] → [destaque ou contraste] → [conclusão de negócio].
                            5. Máximo 3 frases.

                            RESPOSTA:"""

                            resposta_narrador = client.chat.completions.create(
                                model="llama3",
                                messages=[{"role": "user", "content": prompt_narrador}],
                                temperature=0.0
                            )
                            
                            explicacao_texto = resposta_narrador.choices[0].message.content.strip()
                            
                            # Exibe o texto e gera o áudio em memória (sem salvar arquivo físico)
                            st.info(f"🎙️ **Explicação:** {explicacao_texto}")
                            
                            tts = gTTS(text=explicacao_texto, lang='pt', tld='com.br')
                            audio_bytes = BytesIO()
                            tts.write_to_fp(audio_bytes)
                            audio_bytes.seek(0)
                            
                            # Toca o áudio no Streamlit
                            st.audio(audio_bytes, format='audio/mp3', autoplay=True)

                except Exception as erro_execucao:
                    st.error(f"Erro ao processar os dados: {erro_execucao}")

            except Exception as e:
                st.error(f"Erro de conexão com a IA: {e}")