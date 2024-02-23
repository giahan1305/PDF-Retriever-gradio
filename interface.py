import os
import gradio as gr
import fitz  # Thư viện PyMuPDF
from langchain.embeddings.openai import OpenAIEmbeddings
import time
import shutil
import retrievergpt
  

embedding_tool = OpenAIEmbeddings(chunk_size=500,embedding_ctx_length=1024,max_retries=3,model_kwargs={"maxConcurrency" : 2})
local_save = retrievergpt.local_save
vector_save = retrievergpt.vector_save
os.makedirs(os.path.dirname(local_save), exist_ok=True)



##############################################################################################################
# Danh sách để lưu trữ thông tin các file đã đọc
read_files_info = {}
readed_file=""
index_file=0
docs=[]
vector_space=[]
def add_text(history, text):
    history = history + [(text, "text")]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    file_name = file.name.split("\\")[-1]
    local_path = os.path.join(os.getcwd(), local_save)
    local_path = os.path.join(local_path, file_name)
    global read_files_info
    global readed_file
    if file_name not in read_files_info:
        if file.name.lower().endswith('.txt'):
            # Đọc nội dung từ file văn bản
            with open(file.name, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
                history = history + [(file_name,"file" )]  # Đánh dấu là file
                read_files_info[file_name] = text
                readed_file=file_name
                global docs
                global vector_space
                shutil.copy(file.name, local_path)
                retrievergpt.preprocess(embedder=embedding_tool, data_path=local_path, vector_path=vector_save)
        elif file.name.lower().endswith('.pdf'):
            # Đọc nội dung từ file PDF bằng PyMuPDF
            with fitz.open(file.name) as pdf_document:
                text = ""
                for page_number in range(pdf_document.page_count):
                    page = pdf_document[page_number]
                    text += page.get_text()
                history = history + [( file_name,"file")]  # Đánh dấu là file
                read_files_info[file_name] = text
                readed_file=file_name
                shutil.copy(file.name, local_path)
                retrievergpt.preprocess(embedder=embedding_tool, data_path=local_path, vector_path=vector_save)
        else:
            history = history + [("Không hỗ trợ định dạng file trên", None)]
    else:
        history = history + [(file_name, None)]
        if (file_name!=readed_file):
            shutil.copy(file.name, local_path)
            retrievergpt.preprocess(embedder=embedding_tool, data_path=local_path, vector_path=vector_save)
    return history



def bot(history):
    db = retrievergpt.loadVector(embedding_tool)
    response = ""
    # Kiểm tra loại của mục cuối cùng trong history
    last_item = history[-1]
    last_item_type = last_item[1]
    if history:
        if last_item_type == "text":
            if len(read_files_info)==0 and db is None:
                response += "Bạn vui lòng nhập file trước khi đặt câu hỏi\n\n"
            else:
                #answer = query(question=last_item[0],vector_space=vector_space,docs=docs,model=model)
                answer = retrievergpt.dbQuery(question=last_item[0],retriever=db,k=3)
                response+=answer
            
        elif last_item_type == "file":
            response += "File bạn vừa tải lên tôi đã đọc xong\n\n"
        
        elif last_item_type is None:
            if (last_item[0]=="Invalid file format"):
                response+="File bạn vừa nhập không được hỗ trợ"
            else:
                response += "File bạn vừa tải lên tôi đã đọc xong\n\n"
    else:
        response += "Dữ liệu không tìm được"
   
        
   
    history[-1][1] = response
    time.sleep(0.005)
    yield history



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Nhập câu hỏi vào đây và nhấn Enter để trả lời. Hoặc là bấm vào 📁 để upload PDF/TXT",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["pdf", "txt"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":  
    demo.launch()
