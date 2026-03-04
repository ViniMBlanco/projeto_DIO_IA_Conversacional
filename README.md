# 📊 BI Conversacional Multimodal (Powered by Local AI)

Um assistente de Análise de Dados 100% local e multimodal. Este projeto combina o poder de LLMs rodando na sua própria máquina com reconhecimento de voz para transformar arquivos CSV em insights conversacionais, gráficos interativos e narrativas em áudio.



## ✨ Principais Funcionalidades

* **📂 Upload e Limpeza Inteligente:** Aceita arquivos `.csv` e realiza conversão automática de tipos (ex: strings com cifras `$`, `,` para números).
* **📈 Auto-EDA (Análise Exploratória Automática):** Gera instantaneamente estatísticas descritivas, contagem de nulos e as top 10 maiores correlações (usando amostragem inteligente para datasets > 50.000 linhas).
* **🎙️ Interação Multimodal (Speech-to-Text):** Converse com seus dados digitando ou gravando áudio direto no navegador, impulsionado pelo modelo **Whisper** da OpenAI.
* **🧠 Geração de Código Dinâmico:** Utiliza o **Llama 3** (via Ollama) para interpretar perguntas de negócio e gerar código Python/Pandas em tempo real.
* **📊 Visualização Avançada:** Gera e renderiza gráficos interativos automaticamente usando **Plotly Express**.
* **🔊 Narrador Analítico (Text-to-Speech):** A IA analisa os resultados obtidos e gera uma explicação executiva em áudio usando a biblioteca **gTTS**.

---

## 🛠️ Tecnologias Utilizadas

* **Frontend & Framework:** [Streamlit](https://streamlit.io/)
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Plotly Express
* **Inteligência Artificial (LLM):** Llama 3 (executado localmente via [Ollama](https://ollama.com/))
* **Reconhecimento de Voz (ASR):** Whisper (OpenAI) + `streamlit-mic-recorder`
* **Síntese de Voz (TTS):** gTTS (Google Text-to-Speech)
* **Processamento de Áudio:** FFmpeg

---

## ⚙️ Pré-requisitos e Instalação

Para rodar este projeto na sua máquina, você precisará ter o Python instalado, além do Ollama e do FFmpeg.

### 1. Dependências de Sistema
* **Ollama:** Baixe e instale o [Ollama](https://ollama.com/). Após instalar, abra o terminal e rode o comando para baixar o modelo base:
    ```bash
    ollama run llama3
    ```
* **FFmpeg:** Necessário para o Whisper processar o áudio.
    * *No Windows:* Abra o terminal como Administrador e rode: `winget install ffmpeg` (Reinicie o computador/terminal após a instalação).
    * *No Linux/Ubuntu:* `sudo apt update && sudo apt install ffmpeg`
    * *No Mac:* `brew install ffmpeg`

### 2. Configuração do Ambiente Python
Clone este repositório e configure o ambiente virtual:

```bash
# Clone o repositório
git clone [https://github.com/SEU-USUARIO/bi-conversacional-ai.git](https://github.com/SEU-USUARIO/bi-conversacional-ai.git)
cd bi-conversacional-ai

# Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# No Windows:
venv\Scripts\activate
# No Mac/Linux:
source venv/bin/activate

# Instale as bibliotecas necessárias
pip install -r requirements.txt
