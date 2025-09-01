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
# Extração e ajustes de PlantUML
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
    if b < 10:
        return chr(48 + b)
    b -= 10
    if b < 26:
        return chr(65 + b)
    b -= 26
    if b < 26:
        return chr(97 + b)
    b -= 26
    if b == 0:
        return '-'
    if b == 1:
        return '_'
    return '?'


def append3bytes(b1, b2, b3):
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return "".join([
        encode_6bit(c1 & 0x3F),
        encode_6bit(c2 & 0x3F),
        encode_6bit(c3 & 0x3F),
        encode_6bit(c4 & 0x3F),
    ])


def encode_plantuml(text):
    data = zlib.compress(text.encode("utf-8"))[2:-4]
    res = ""
    i = 0
    while i < len(data):
        b1 = data[i]
        b2 = data[i + 1] if i + 1 < len(data) else 0
        b3 = data[i + 2] if i + 2 < len(data) else 0
        res += append3bytes(b1, b2, b3)
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

def _mk_llm(temp=0.25, max_tokens=2200):
    return GoogleGenAI(
        model="gemini-2.5-flash",
        temperature=temp,
        max_output_tokens=max_tokens,
    )


def generate_initial_doc_gemini(text):
    llm = _mk_llm(temp=0.3, max_tokens=1500)
    prompt_content = f"""Você é um(a) engenheiro(a) de software.
Gere um **documento inicial e resumido** com base no texto fornecido.
Esse documento deve dar ao desenvolvedor um **entendimento primário do projeto**, cobrindo:

- Nome do Projeto (provisório)
- Objetivo resumido
- Contexto geral do problema
- Principais requisitos funcionais (em poucas linhas)
- Principais requisitos não funcionais (em poucas linhas)
- Observações relevantes

=== TEXTO DE REFERÊNCIA ===
{text}
"""
    response = llm.complete(prompt_content)
    return response.text


def generate_doc_gemini(text):
    """Gera o documento completo com foco arquitetural e RNFs mapeados a componentes."""
    llm = _mk_llm(temp=0.25, max_tokens=2600)
    prompt_content = f"""Você é um **Arquiteto(a) de Software e Soluções** responsável por elaborar **documentação de arquitetura técnica**.
Sua missão é transformar o texto abaixo em um documento **padronizado e claro**, que apoie **desenvolvedores, analistas e stakeholders** a compreenderem a solução proposta sob a ótica arquitetural.

Siga o formato abaixo como referência **obrigatória**:

=================== EXEMPLO DE FORMATO ===================

**Documento de Arquitetura de Software e Soluções – [Nome do Sistema]**
Data: [Data Atual]
Autor: Arquiteto(a) de Software e Soluções
Versão: 1.0

**1) Objetivo**
Explique de forma clara o propósito do sistema, seu escopo e a visão arquitetural de alto nível (por ex.: foco em escalabilidade, segurança, performance, integração, observabilidade).

**2) Contexto e Requisitos**
- **Contexto**: Explique o problema/necessidade que o sistema resolve.
- **Requisitos Funcionais (RF)**: Liste numerada (RF001, RF002, …).
- **Requisitos Não Funcionais (RNF)**:
  - Liste numerada (RNF001, RNF002, …).
  - **Associe explicitamente cada RNF aos componentes relevantes do diagrama** (ex.: ComponentDb, ContainerQueue, API Gateway, Service X). Use a sintaxe: `RNF00X – [Categoria] – Componentes afetados: [ComponenteA, ComponenteB] – Descrição objetiva`.
  - Cubra, quando aplicável: disponibilidade, recuperação de desastre, escalabilidade, latência, throughput, segurança (authN/authZ, criptografia em trânsito e em repouso), confidencialidade/integridade, resiliência, observabilidade (logs, métricas, tracing), custo, conformidade, suporte/operabilidade.

**3) Arquitetura e Diagrama de Componentes (C4-PlantUML)**
Inclua um diagrama C4-PlantUML **estritamente** na notação C4, contendo:
- `Person` (usuários/atores)
- `System_Boundary` (escopo do sistema)
- `Component` (módulos internos)
- `Rel` (relações)
❌ Não utilize `rectangle`, `note`, `legend` ou sintaxes UML clássicas.
❗ Para bancos de dados, use **ComponentDb** ou **ComponentDb_Ext**.
❗ Para filas/mensageria, use **ContainerQueue** ou **ContainerQueue_Ext**.
Após o diagrama, descreva **cada componente** com: responsabilidades, dependências, dados tratados, e **quais RNFs o impactam** (referencie os IDs RNF00X).

**4) Integrações Externas e Contratos em Alto Nível**
Liste sistemas externos e descreva protocolos, padrões de integração (REST, gRPC, eventos), formatos (JSON, Avro, Protobuf), versionamento, e objetivos.

**5) Decisões Arquiteturais Relevantes**
Registre decisões e trade-offs (por ex.: microsserviços vs. monólito; sync vs. async; banco relacional vs. NoSQL; fila vs. streaming; segurança). Se possível, use formato ADR curto: Contexto → Decisão → Consequências.

**6) Critérios de Aceite e Próximos Passos**
- Critérios mínimos de aceite.
- Próximos passos para evolução da arquitetura e implementação.

=================== FIM DO EXEMPLO ===================

=== TEXTO BASE ===
{text}
"""
    response = llm.complete(prompt_content)
    return response.text


def generate_prototype_gemini(user_story):
    llm = _mk_llm(temp=0.35, max_tokens=1800)
    prompt_content = f"""Você é um(a) engenheiro(a) de software.
Com base na **história do usuário** abaixo, gere um **protótipo inicial** que ajude a equipe a visualizar a solução.
O protótipo pode conter:
- Uma breve descrição do fluxo de interação.
- Um diagrama simples em C4-PlantUML (System_Boundary, Person, Component, ContainerQueue, ComponentDb, etc.).
- Observações sobre possíveis telas ou APIs envolvidas.

=== HISTÓRIA DO USUÁRIO ===
{user_story}
"""
    response = llm.complete(prompt_content)
    return response.text


# =============================
# App Streamlit (UI)
# =============================

st.set_page_config(page_title="Gerador de Artefatos de Projeto", layout="wide")
st.title("🎙️ Gerador de Artefatos de Projeto")

st.markdown(
    "Escolha abaixo como fornecer sua entrada e qual tipo de documento deseja gerar."
)

# Entrada
modo = st.radio(
    "📥 Como deseja fornecer o conteúdo?",
    ["Upload de Áudio", "Digitar Texto"],
)

texto = None

if modo == "Upload de Áudio":
    audio_file = st.file_uploader(
        "Carregue o arquivo de áudio (mp3, wav, opus, ogg, m4a):",
        type=["mp3", "wav", "opus", "ogg", "m4a"],
    )

    if audio_file and st.button("Transcrever Áudio"):
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
    texto_input = st.text_area("✍️ Digite ou cole o texto para análise:", height=220)
    if st.button("Usar Texto Digitado"):
        texto = texto_input

# Saída
TipoSaida = [
    "Documento Inicial",
    "Documento Completo de Arquitetura",
    "Protótipo Inicial",
    "Gerar Todos",
]

tipo_saida = st.selectbox("📄 Qual documento você deseja gerar?", TipoSaida)

# Geração
if texto:
    resultados = {}

    if tipo_saida in ("Documento Inicial", "Gerar Todos"):
        with st.spinner("📝 Gerando documento inicial..."):
            resultados["Documento Inicial"] = generate_initial_doc_gemini(texto)

    if tipo_saida in ("Documento Completo de Arquitetura", "Gerar Todos"):
        with st.spinner("📄 Gerando documentação completa..."):
            resultados["Documento Completo"] = generate_doc_gemini(texto)

    if tipo_saida in ("Protótipo Inicial", "Gerar Todos"):
        with st.spinner("🎨 Gerando protótipo inicial..."):
            resultados["Protótipo"] = generate_prototype_gemini(texto)

    # Exibição e diagramas
    for titulo, resultado in resultados.items():
        st.subheader(f"📌 {titulo}")
        st.markdown(resultado)

        with st.spinner("📊 Montando diagrama, se aplicável..."):
            plantuml_code = extract_plantuml_code(resultado)
            if plantuml_code:
                svg_content = plantuml_get_diagram(plantuml_code, "svg")
                if svg_content:
                    st.components.v1.html(svg_content, height=520, scrolling=True)

    # Persistência no estado
    st.session_state["transcricao"] = texto
    st.session_state["documentos"] = resultados

# Download ZIP
if "documentos" in st.session_state and st.session_state["documentos"]:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Transcrição
        transcricao_path = os.path.join(tmpdir, "transcricao.txt")
        with open(transcricao_path, "w", encoding="utf-8") as f:
            f.write(st.session_state["transcricao"])

        # Documentos
        for nome, conteudo in st.session_state["documentos"].items():
            doc_filename = nome.lower().replace(" ", "_") + ".md"
            doc_path = os.path.join(tmpdir, doc_filename)
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(conteudo)

        # ZIP
        zip_path = os.path.join(tmpdir, "artefatos_projeto.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(transcricao_path, arcname="transcricao.txt")
            for nome in st.session_state["documentos"].keys():
                doc_filename = nome.lower().replace(" ", "_") + ".md"
                zf.write(os.path.join(tmpdir, doc_filename), arcname=doc_filename)

        with open(zip_path, "rb") as f:
            st.download_button(
                "📥 Baixar Todos os Artefatos",
                data=f,
                file_name="artefatos_projeto.zip",
                mime="application/zip",
            )
