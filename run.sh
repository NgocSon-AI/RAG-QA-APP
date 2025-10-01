#!/bin/bash

# Dừng khi có lỗi
set -e
# Tao moi

# Chạy ứng dụng Streamlit
streamlit run app.py --server.port 8501
