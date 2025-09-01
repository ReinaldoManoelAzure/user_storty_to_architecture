# Imagem base com Python
FROM python:3.11-slim

# Evita prompts interativos do apt
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema (ffmpeg é essencial para whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Porta padrão do Streamlit
EXPOSE 8501

# Comando de entrada
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
