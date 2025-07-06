# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain_community.chat_models import ChatOllama
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# import csv
# from langchain.schema import Document
# import os

# # loader=PyPDFLoader("C:\\Users\\Mahesh\\Downloads\\Courses_Schedule.pdf")
# # load_documents=loader.load()
# # CS_doc=CSVLoader("C:\\Users\\Mahesh\\Downloads\\CS.csv")
# # MIS_doc=CSVLoader("C:\\Users\\Mahesh\\Downloads\\Book2.csv")
# # load_documents=CS_doc.load()+MIS_doc.load()
# # print(load_documents)

# def load_csv_as_documents(file_path):
#     documents = []
#     with open(file_path, encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             if not any(row.values()):
#                 continue  # Skip empty rows

#             # Format the row into readable text
#             content = "\n".join(f"{key.strip()}: {value.strip()}" for key, value in row.items() if value.strip())
#             documents.append(Document(page_content=content))
#     return documents

# # Use it:
# cs_docs = load_csv_as_documents("C:\\Users\\Mahesh\\Downloads\\CS.csv")
# mis_docs = load_csv_as_documents("C:\\Users\\Mahesh\\Downloads\\Book2.csv")

# all_docs = cs_docs + mis_docs

# Splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=100)
# documents=Splitter.split_documents(all_docs)


# # print(documents)
# embeddings=OllamaEmbeddings(model="llama3")

# persist_dir = "chroma_db_persist"

# if os.path.exists(persist_dir) and os.listdir(persist_dir):
#     #  Load existing vector DB
#     vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
# else:
#     #  Create it for the first time
#     vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
#     vector_store.persist()

# retriever=vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k":20}
# )


# # results = vector_store.similarity_search("Computer science courses")
# # for doc in results:
# #     print(doc.page_content)

# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True,
#     output_key="answer"
# )

# llm=ChatOllama(model="llama3")


# # docs = retriever.get_relevant_documents("Computer science")
# # for doc in docs:
# #     print("\n--- Document Chunk ---")
# #     print(doc.page_content)


# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     return_source_documents=True
# )

# while True:
#     print("input: ")
#     query = input("You: ")
#     if query.lower() in ["exit", "quit"]:
#         print("👋 Goodbye!")
#         break

#     result = qa_chain.invoke({"question": query})
#     print(f"\nBot: {result['answer']}\n")



# ------------------------------------------------------------------------------------------------------------------------------------------





import os
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# '''This is to get Content from URL and save as a file'''
# def get_page_content(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     return soup.get_text()
# urls = [
#     "https://catalog.lamar.edu/college-arts-sciences/computer-science/computer-science-ms/#text"
# ]

# documents = [get_page_content(url) for url in urls]

# with open("C:\\Users\\Mahesh\Documents\\Rag_CS_Documents\\overview.txt", "w", encoding="utf-8") as f:
#     f.write(documents[0])
# docs = [Document(page_content=doc) for doc in documents]


def degree_audit():
    degree_audit = PyPDFLoader("C:\\Users\\Mahesh\\Downloads\\Degree_Audit.pdf")
    courses_list=PyPDFLoader("C:\\Users\\Mahesh\\Downloads\\Registration.pdf")
    load_courses_list=courses_list.load()
    load_degree_audit=degree_audit.load()
    for doc in load_degree_audit:
        doc.metadata["source"]="degree_audit"
        doc.page_content=f"this is related to degree_audit"+doc.page_content
    for doc in load_courses_list:
        doc.metadata["source"]="Fall_courses"
        doc.page_content=f"this is related to Fall available courses"+doc.page_content
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(load_degree_audit)
    chunks.extend(splitter.split_documents(load_courses_list))
    return chunks

# --------------------------------------------------------------------------------------------------------------------------------

'''The function adds metadata and merge all documents into a list'''
def document_parsing(folder_path):
    documents=[]

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            loader=TextLoader(file_path,encoding='utf-8')
            docs=loader.load()
            for doc in docs:
                doc.metadata["source"]=filename
                doc.page_content=f"This is a {filename}" + doc.page_content
            documents.extend(docs)
    return documents

folder_path = "Computer Science"
docs= document_parsing(folder_path=folder_path)
docs.extend(degree_audit())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory="./nomic_data"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    #  Load existing vector DB
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    #  Create it for the first time
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    vector_store.persist()


retriever = vector_store.as_retriever(search_kwargs={"k": 4})

llm=ChatOllama(model="llama3")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True  # optional: shows which doc answers came from
)

# while True:
#     print("input: ")
#     query = input("You: ")
#     if query.lower() in ["exit", "quit"]:
#         print("👋 Goodbye!")
#         break

#     result = qa_chain.invoke({"question": query})
#     print(f"\nBot: {result['answer']}\n")

while True:
    print("input: ")
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    retrieved_docs = retriever.get_relevant_documents(query)
# Print the retrieved chunks
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Retrieved Chunk {i+1} ---")
        print(doc.page_content)
        print("-----------------------------------------------------------------------------------------------------------------")

    result = qa_chain.invoke({"question": query})
    print(f"\nBot: {result['answer']}\n")