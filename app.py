import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
#from logger import logger
from rag_utils import read_file, split_text, get_embedding_model, build_vectorstore
import time
# GROQ SDK
from groq import Groq

from sklearn.metrics.pairwise import cosine_similarity


st.sidebar.title("Cấu hình mô hình")

CHAT_MODELS = {
    "Llama 3.1 8B Instant": {"type": "groq", "model_name": "llama-3.1-8b-instant"},
    "meta-llama/llama-guard-4-12b": {"type": "groq", "model_name": "meta-llama/llama-guard-4-12b"},
    "gemma2-9b-it": {"type": "groq", "model_name": "gemma2-9b-it"},
    "moonshotai/kimi-k2-instruct-0905": {"type": "groq", "model_name": "moonshotai/kimi-k2-instruct-0905"},
    "openai/gpt-oss-20b": {"type": "groq", "model_name": "openai/gpt-oss-20b"},
    "groq/compound": {"type": "groq", "model_name": "groq/compound"},
}

selected_chat_model = st.sidebar.selectbox("Chọn mô hình chat:", list(CHAT_MODELS.keys()))

EMBEDDING_MODELS = {
    "(offline) multilingual-e5-small": {"type": "sentence-transformer", "model_name": "intfloat/multilingual-e5-small"},
    "(offline) multilingual-MiniLM-L12-v2": {"type": "sentence-transformer", "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
    "(offline) distiluse-base-multilingual-cased-v1": {"type": "sentence-transformer", "model_name": "sentence-transformers/distiluse-base-multilingual-cased-v1"},
    "(offline) multilingual-e5-base": {"type": "sentence-transformer", "model_name": "intfloat/multilingual-e5-base"}, 
    "OpenAIEmbeddings": {"type": "openai"},
}
selected_embedding_model = st.sidebar.selectbox("Chọn mô hình embedding:", list(EMBEDDING_MODELS.keys()))

USE_FAISS = True

#from logger import logger

uploaded_files = st.file_uploader(
    "Upload nhiều file cùng lúc", 
    type=["pdf","docx","txt","csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    #logger.info({"event": "file_upload", "num_files": len(uploaded_files),
    #             "files": [f.name for f in uploaded_files]})
    
    all_texts = []
    for file in uploaded_files:
        text = read_file(file)
        if text:
            chunks = split_text(text)
            if chunks:
                all_texts.extend(chunks)
    st.success(f"Đã chia văn bản thành {len(all_texts)} đoạn!")
    #logger.info({"event": "text_split", "num_chunks": len(all_texts)})


    embed_info = EMBEDDING_MODELS[selected_embedding_model]
    if embed_info["type"] == "openai":
        embedding_model = get_embedding_model(openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        embedding_model = get_embedding_model(model_name=embed_info["model_name"])

    vectordb = build_vectorstore(all_texts, embedding_model, use_faiss=USE_FAISS)
    st.session_state['vectordb'] = vectordb
    st.success("✅ Vector store đã được tạo thành công!")


# ---------------- QA ----------------
if 'vectordb' in st.session_state:
    query = st.text_input("Nhập câu hỏi của bạn bằng tiếng Việt:")

    if query:
        # --- Initialize GROQ client ---
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=GROQ_API_KEY)

        #logger.info({"event": "query_received", "query": query})

        # --- Top-k retrieval ---
        
        docs_and_scores = st.session_state['vectordb'].similarity_search_with_score(query, k=5)
        # Ghép context từ văn bản
        context_text = "\n\n".join([doc.page_content for doc, score in docs_and_scores])

        # retriever = st.session_state['vectordb'].as_retriever(search_kwargs={"k":5})
        
        #relevant_chunks = docs_and_scores.get_relevant_documents(query)
        
        #logger.info({"event": "retrieval_done", "num_docs": len(docs_and_scores)})
        
        #context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])

        # --- Prompt ---
        prompt = f"""
Bạn là một trợ lý AI am hiểu sâu rộng về trí tuệ nhân tạo. 
Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên tài liệu được cung cấp. 
Nếu tài liệu không có thông tin liên quan, hãy trả lời đúng nguyên văn:
"tôi không biết, kiến thức về câu hỏi không có trong tài liệu".

Yêu cầu khi trả lời:
- Ngắn gọn, súc tích.
- Trình bày theo gạch đầu dòng từng ý.
- Chỉ dựa trên thông tin có trong tài liệu.

Ví dụ minh họa:
"Supervised learning (Học có giám sát)" là:
- Một kỹ thuật học máy.
- Thuật toán được huấn luyện trên dữ liệu đã gán nhãn (đầu vào và đầu ra).
- Giúp dự đoán kết quả cho dữ liệu mới.

"CNN" là:
- Convolutional Neural Network (Mạng Nơ-ron Tích chập).
- Mô hình học sâu cho xử lý ảnh và thị giác máy tính.
- Dùng các tầng tích chập để trích xuất đặc trưng.

“Prompt” là cách bạn giao tiếp với AI, và việc tạo ra một prompt hiệu quả đóng vai trò quan trọng trong việc đạt được kết quả mong muốn. Một cấu trúc chuẩn để tạo prompt:
- Ngữ cảnh: Mô tả chủ đề hoặc vấn đề bạn cần AI hỗ trợ.
- Hướng dẫn: Đưa ra yêu cầu cụ thể.
- Dữ liệu đầu vào: Cung cấp thông tin hoặc dữ liệu cho AI sử dụng.
- Kết quả mong muốn: Chỉ định rõ kết quả bạn kỳ vọng.
---

Văn bản tham khảo:
{context_text}

Câu hỏi:
{query}
        """

        # --- Gọi GROQ API ---
        response = client.chat.completions.create(
            model=CHAT_MODELS[selected_chat_model]["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=300
        )

        answer = response.choices[0].message.content
        st.markdown(f"**Trả lời:** {answer}")
        #logger.info({"event": "response_sent", "answer": answer})

        st.markdown("### 🔹 Tài liệu tham khảo")
        for i, (doc, score) in enumerate(docs_and_scores, 1):
            source = doc.metadata.get("source", "Unknown")  # lấy metadata nguồn
            with st.expander(f"🔹 Chunk {i} (Nguồn: {source}, Độ tương đồng: {score:.4f})"):
                st.write(doc.page_content)
