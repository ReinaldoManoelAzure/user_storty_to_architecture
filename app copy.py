import streamlit as st
import ffmpeg
import whisper
import os
import re
import tempfile
import zipfile
import zlib
import requests
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

# =============================
# Utilitários de mídia
# =============================
def convert_to_mp3(input_file, ext):
    output_path = "temp.mp3"
    (
        ffmpeg
        .input(input_file)
        .output(output_path, format='mp3', acodec='libmp3lame')
        .run(quiet=True, overwrite_output=True)
    )
    return output_path

def transcribe_audio(path):
    model = whisper.load_model("base")
    result = model.transcribe(path, language="pt")
    return result["text"]

# =============================
# Extração PlantUML
# =============================
def extract_plantuml_code(text: str) -> str | None:
    match = re.search(r'@startuml.*?@enduml', text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    code = match.group(0).strip()

    # Ajuste de componentes de banco de dados e filas
    code = re.sub(r'(?<![A-Za-z])(Database|Db)(?![A-Za-z])', 'ComponentDb', code, flags=re.IGNORECASE)
    code = re.sub(r'(?<![A-Za-z])(Database_Ext|Db_Ext)(?![A-Za-z])', 'ComponentDb_Ext', code, flags=re.IGNORECASE)
    code = re.sub(r'(?<![A-Za-z])(Queue)(?![A-Za-z])', 'ContainerQueue', code, flags=re.IGNORECASE)
    code = re.sub(r'(?<![A-Za-z])(Queue_Ext)(?![A-Za-z])', 'ContainerQueue_Ext', code, flags=re.IGNORECASE)

    return code

# =============================
# Encoder oficial PlantUML
# =============================
def encode_6bit(b):
    if b < 10: return chr(48 + b)
    b -= 10
    if b < 26: return chr(65 + b)
    b -= 26
    if b < 26: return chr(97 + b)
    b -= 26
    if b == 0: return '-'
    if b == 1: return '_'
    return '?'

def append3bytes(b1, b2, b3):
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return "".join([encode_6bit(c1 & 0x3F), encode_6bit(c2 & 0x3F),
                    encode_6bit(c3 & 0x3F), encode_6bit(c4 & 0x3F)])

def encode_plantuml(text):
    data = zlib.compress(text.encode("utf-8"))[2:-4]
    res = ""
    i = 0
    while i < len(data):
        b1 = data[i]
        b2 = data[i+1] if i+1 < len(data) else 0
        b3 = data[i+2] if i+2 < len(data) else 0
        res += append3bytes(b1,b2,b3)
        i += 3
    return res

def plantuml_get_diagram(plantuml_code, fmt="svg"):
    """Gera diagrama PlantUML em SVG ou PNG via servidor público."""
    server_url = f"http://www.plantuml.com/plantuml/{fmt}/"
    encoded = encode_plantuml(plantuml_code)
    url = server_url + encoded
    response = requests.get(url)
    if response.status_code == 200:
        return response.content if fmt == "png" else response.text
    else:
        st.warning(f"Não foi possível gerar o diagrama PlantUML em {fmt}.")
        return None

# =============================
# LLM - Gemini via GoogleGenAI
# =============================
def generate_doc_gemini(text):
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        temperature=0.25,
        max_output_tokens=2200
    )

    prompt_content = f"""Você é um(a) engenheiro(a) de software responsável por criar **documentação de arquitetura**.
Transforme o texto abaixo em um documento no formato **padronizado**, inspirado no exemplo fornecido:

=================== EXEMPLO DE FORMATO ===================

Documento de Arquitetura de Software - [Nome do Sistema]
Data: [Data Atual]
Autor: Engenheiro(a) de Software
Versão: 1.0

1) Objetivo: 
Texto claro e objetivo explicando o propósito do sistema e a abordagem de arquitetura de alto nível.

2) Contexto e Requisitos
- Contexto: Explicação breve do problema/necessidade do sistema.
- Requisitos Funcionais (RF): Lista numerada de requisitos funcionais (RF001, RF002, ...).
- Requisitos Não Funcionais (RNF): Lista numerada de requisitos não funcionais (RNF001, RNF002, ...).

3) Diagrama de Componentes (C4-PlantUML)
Inclua um diagrama PlantUML seguindo **estritamente** a notação C4-PlantUML, com:
- Person
- System_Boundary
- Component(s)
- Rel
❌ Não use `rectangle`, `note`, `legend` ou sintaxes UML clássicas.
❗ Para bancos de dados use **ComponentDb ou ComponentDb_Ext**.
❗ Para filas use **ContainerQueue ou ContainerQueue_Ext**.
Após o diagrama, descreva cada componente em texto.

4) Integrações Externas e Contratos em Alto Nível
Liste integrações externas (se houver), descrevendo protocolos, formatos de dados e propósito.

5) Critérios de Aceite e Próximos Passos
- Critérios de Aceite mínimos
- Próximos passos

=================== FIM DO EXEMPLO ===================

=== TEXTO DA REUNIÃO ===
{text}
"""
    response = llm.complete(prompt_content)
    return response.text

# =============================
# App Streamlit
# =============================
st.set_page_config(page_title="Documento de Arquitetura", layout="wide")
st.title("🎙️ Documento de Arquitetura")

# Escolha do modo de entrada
modo = st.radio(
    "Escolha como fornecer a entrada:",
    ["Upload de Áudio", "Digitar Texto"]
)

texto = None

if modo == "Upload de Áudio":
    audio_file = st.file_uploader(
        "Carregue o arquivo de áudio (mp3, wav, opus, ogg, m4a):",
        type=["mp3", "wav", "opus", "ogg", "m4a"]
    )

    if audio_file and st.button("Gerar Documento"):
        ext = audio_file.name.split('.')[-1].lower()
        if ext != "mp3":
            temp_path = f"temp_input.{ext}"
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            mp3_path = convert_to_mp3(temp_path, ext)
            os.remove(temp_path)
        else:
            mp3_path = "temp.mp3"
            with open(mp3_path, "wb") as f:
                f.write(audio_file.getbuffer())

        with st.spinner("🔎 Transcrevendo áudio..."):
            texto = transcribe_audio(mp3_path)

elif modo == "Digitar Texto":
    texto_input = st.text_area("Digite ou cole o texto para análise:", height=200)
    if st.button("Gerar Documento"):
        texto = texto_input

# Se houver texto (transcrição ou digitado), gera o documento
if texto:
    with st.spinner("🤖 Gerando documento com Gemini..."):
        doc_text = generate_doc_gemini(texto)

    with st.spinner("📊 Montando diagrama..."):
        plantuml_code = extract_plantuml_code(doc_text)
        svg_content, png_content = None, None
        if plantuml_code:
            svg_content = plantuml_get_diagram(plantuml_code, "svg")
            png_content = plantuml_get_diagram(plantuml_code, "png")

    # Persistir no estado
    st.session_state["transcricao"] = texto
    st.session_state["documento"] = doc_text
    st.session_state["plantuml"] = plantuml_code
    st.session_state["svg"] = svg_content
    # st.session_state["png"] = png_content

# =============================
# Exibição persistente
# =============================
if "transcricao" in st.session_state:
    st.subheader("📝 Texto Transcrito")
    st.text_area("Resultado da transcrição:", st.session_state["transcricao"], height=200)

if "documento" in st.session_state:
    st.subheader("📄 Documento Técnico")
    st.markdown(st.session_state["documento"])

if "svg" in st.session_state and st.session_state["svg"]:
    st.subheader("📊 Diagrama de Componentes (SVG)")
    st.components.v1.html(st.session_state["svg"], height=600, scrolling=True)

# =============================
# Download ZIP
# =============================
if "documento" in st.session_state and st.session_state["documento"]:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Transcrição
        transcricao_path = os.path.join(tmpdir, "transcricao.txt")
        with open(transcricao_path, "w", encoding="utf-8") as f:
            f.write(st.session_state["transcricao"])

        # Documento + referência ao PNG
        doc_path = os.path.join(tmpdir, "documentacao.md")
        doc_text_final = st.session_state["documento"]
        if st.session_state.get("png"):
            doc_text_final += "\n\n---\n\n### Diagrama de Componentes\n"
            doc_text_final += "![Diagrama de Componentes](diagrama.png)\n"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_text_final)

        # PlantUML + imagens
        if st.session_state.get("plantuml"):
            puml_path = os.path.join(tmpdir, "diagrama.puml")
            with open(puml_path, "w", encoding="utf-8") as f:
                f.write(st.session_state["plantuml"])
        if st.session_state.get("svg"):
            svg_path = os.path.join(tmpdir, "diagrama.svg")
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(st.session_state["svg"])
        if st.session_state.get("png"):
            png_path = os.path.join(tmpdir, "diagrama.png")
            with open(png_path, "wb") as f:
                f.write(st.session_state["png"])

        # ZIP
        zip_path = os.path.join(tmpdir, "artefatos_arquitetura.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(transcricao_path, arcname="transcricao.txt")
            zf.write(doc_path, arcname="documentacao.md")
            if st.session_state.get("plantuml"):
                zf.write(puml_path, arcname="diagrama.puml")
            if st.session_state.get("svg"):
                zf.write(svg_path, arcname="diagrama.svg")

        with open(zip_path, "rb") as f:
            st.download_button(
                "📥 Baixar Pacote Completo",
                data=f,
                file_name="artefatos_arquitetura.zip",
                mime="application/zip"
            )
