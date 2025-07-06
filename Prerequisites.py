import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

"""
few_shot_prompt = 

"""
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


from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
def degree_audit():
    degree_docs = []
    course_docs = []

    # Load PDF files
    degree_loader = TextLoader("C:\\Users\\Mahesh\\Documents\\Degree_Audit.txt")
    course_loader = CSVLoader("C:\\Users\\Mahesh\\Downloads\\CS.csv")

    degree_pages = degree_loader.load()
    course_pages = course_loader.load()

    # Tag and prepare degree documents
    for doc in degree_pages:
        doc.metadata["source"] = "degree_audit"
        doc.page_content = "This is related to degree audit: " + doc.page_content
        degree_docs.append(doc)

    # Tag and prepare course documents
    for doc in course_pages:
        doc.metadata["source"] = " Fall Course Registration"
        doc.metadata["category"]="Course Registration"
        doc.metadata["semester"]="Fall 2025"
        doc.page_content = "This is related to fall available courses: " + doc.page_content
        course_docs.append(doc)

    return course_docs  # return a flat list of Document objects   ***********degree_docs +


def webBaseLoader():
    urls=["https://catalog.lamar.edu/college-arts-sciences/computer-science/computer-science-ms/#text",
          "https://www.lamar.edu/financial-aid/withdrawing-and-the-60-dates.html",
        # "https://www.lamar.edu/arts-sciences/computer-science/degrees/graduate/degree-requirements.html",
         "https://catalog.lamar.edu/college-arts-sciences/computer-science/computer-science-ms/#requiredcoursestext",
         "https://www.lamar.edu/arts-sciences/computer-science/degrees/graduate/grad-course-descriptions.html",
         ]
    loader=UnstructuredURLLoader(urls)
    docs= loader.load()
    return docs


clarify_prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Rewrite this question for clarity and completeness: {question}"
)


system_msg = SystemMessagePromptTemplate.from_template(
    "You are an expert Academic Advisor for Computer Science Graduate Students, Who helps in student course registration." \
    " Only answer based on the given documents. Do not guess."
)
human_msg = HumanMessagePromptTemplate.from_template(
    "Question: {question}\n\nDocuments:\n{context}"
)

final_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

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

# folder_path = "Rag_CS_Documents"
# docs= document_parsing(folder_path=folder_path)
docs= webBaseLoader()
docs.extend(degree_audit())
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
persist_directory="./rag_data"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    #  Load existing vector DB
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    #  Create it for the first time
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    vector_store.persist()


retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 12})

llm=ChatOllama(model="llama3",temperature=0.4)
# llm = ChatOpenAI(model_name="gpt-4")

clarify_chain = llm | clarify_prompt

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    # condense_question_llm=clarify_chain,  # Uses your custom clarification prompt
    combine_docs_chain_kwargs={"prompt": final_prompt},
    return_source_documents=True  # optional: shows which doc answers came from
)

while True:
    print("input: ")
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    result = qa_chain.invoke({"question": query})
    print(f"\nBot: {result['answer']}\n")


# retrieved_docs = retriever.get_relevant_documents("list of all courses available to register for fall 2025")
# # Print the retrieved chunks
# for i, doc in enumerate(retrieved_docs):
#     print(f"\n--- Retrieved Chunk {i+1} ---")
#     print(doc.page_content)
#     print("-----------------------------------------------------------------------------------------------------------------")
