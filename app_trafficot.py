import streamlit as st
from streamlit_chat import message
import os
from backend import index_export

os.environ["OPENAI_API_KEY"] = "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"
st.header("Trafficot - Trợ lý Giao thông AI 🗺️🚗")
st.info("Xem thêm thông tin tại (http://giaothong.hochiminhcity.gov.vn/)", icon="📃")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Hãy cho tôi biết bạn cần gì? 😁 "}
    ]


@st.cache_resource(show_spinner=False)
def load_response(message):
    return index_export(message)


# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
if prompt := st.chat_input("Câu hỏi của bạn"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Đang tải..."):
            response = load_response(message["content"])
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
