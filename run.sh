#!/bin/bash

# Dừng khi có lỗi
set -e
uv venv .venv
# Kích hoạt môi trường ảo (sửa lại đường dẫn nếu khác)
source .venv/bin/activate
uv sync
# Chọn model embeddings và model chat (sửa theo nhu cầu)
# export EMBEDDING_MODEL="intfloat/multilingual-e5-base"
# export CHAT_MODEL="llama-3.1-70b-versatile"

# Chạy ứng dụng Streamlit
streamlit run app.py --server.port 8501
