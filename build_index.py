from fastembed import TextEmbedding
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings

loader = TextLoader("data/knowledge.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("faiss_index")

print(f"✅ FAISS index built with {len(chunks)} chunks")