API_KEY = ""
import os
import glob
import time
from PyPDF2 import PdfReader # importing required modules
from underthesea import word_tokenize;
from underthesea import text_normalize;
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
os.environ["OPENAI_API_KEY"]  = API_KEY
from langchain.vectorstores import FAISS
import openai
import fitz
local_save = "uploaded_files\\"
vector_save = "vector_db\\"

def loadAll(embedder):
    currPath = os.getcwd()
    filesName = glob.glob(os.path.join(currPath,'vector_*'))
    print(filesName)
    if len(filesName) == 0:
        return None
    db = FAISS.load_local(filesName[0],embedder)
    for i in range(1,len(filesName)):
        doc = FAISS.load_local(filesName[i],embedder)
        db.merge_from(doc)
    return db
    


###############################################################

def preprocess(embedder, data_path="", vector_path=""):
    # Sử dụng thư viện PyMuPDF đọc file do nhanh hơn PyPDF2
    with fitz.open(data_path) as pdf_document:
        text = ''
        for page_number in range(pdf_document.page_count):
            # Tokenize
            page = pdf_document[page_number]
            
            text += word_tokenize(text_normalize(page.get_text()), format="text")
        textSplitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        docs = textSplitter.split_text(text)
        #Ghi vector ra file "vector_[file_name]"
        iterator = 0
        vectors = []
        start = time.time()
        
        while iterator < len(docs):
            text_embeddings = None
            text_embedding_pairs = None
            if iterator + 300 < len(docs):
                text_embeddings = embedder.embed_documents(docs[iterator : (iterator + 300) ],chunk_size=500)
                text_embedding_pairs = zip(docs[iterator :], text_embeddings)
            else:
                text_embeddings = embedder.embed_documents(docs[iterator :],chunk_size=500)
                text_embedding_pairs = zip(docs[iterator :], text_embeddings)
                vectors.extend(text_embedding_pairs)
            if iterator % 3 == 0:
                end = time.time()
                time.sleep(60 - (end - start) + 1) # 1 phút gọi 3 lần
                start = time.time()
            iterator += 300
        db = FAISS.from_embeddings(vectors, embedder)
        save_name = data_path.split('\\')[-1]
        vector_path = os.path.join(vector_path, f'vector_{save_name}')
        db.save_local(vector_path)

def loadVector(embedder):
    # Địa chỉ folder hiện tại
    currPath = os.getcwd()
    # Lấy các file vector đã lưu
    os.makedirs(os.path.join(currPath,'uploaded_files'),exist_ok=True)
    #print(os.path.join(currPath,'uploaded_files'))
    filesName = glob.glob(os.path.join(currPath,vector_save,'vector_*'))
    # Không có thì out
    if len(filesName) == 0:
        return None
    # Load các file vào database
    db = FAISS.load_local(filesName[0],embedder)
    for i in range(1,len(filesName)):
        doc = FAISS.load_local(filesName[i],embedder)
        db.merge_from(doc)
    return db

def dbQuery(question, retriever, k=10):
    if (retriever == None):
        return "Hệ thống hiện tại không tìm thấy bất cứ file nào. Vui lòng thử upload file lại."
    docs = retriever.similarity_search_with_score(query=question,k=k)
    context = ''
    for doc in docs:
        context = context + '. ' + doc[0].page_content
    print(context)
    respond =  openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [
        {"role" : "system",
        "content" : f"Bạn là một hỗ trợ viên trả lời câu hỏi sử dụng bối cảnh đã được cung cấp tại tin nhắn đầu tiên, bạn sẽ trả lời bằng tiếng việt và nội dung câu trả lời chỉ có thể nằm trong nội dung của bối cảnh, bạn có thể tái cấu trúc câu cho câu trả lời đúng về ngữ nghĩa và ngữ pháp. Tuyệt đối không trả lời bằng kiến thức của bạn và nếu như bối cảnh không chứa nội dung cần trả lời từ câu hỏi thì hãy nói không trả lời được tuyệt đối không sáng tạo."
        },
        {"role" : "user",
         "content": f'Bối cảnh là : {context}'
        },
        {"role" : "user",
        "content" : f'Câu hỏi là : {question}.'}
        ],
        temperature = 0,
        stop=["\n"]
    )
    return respond.choices[0]["message"]["content"]
