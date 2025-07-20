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
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader




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


def print_prompt(final_prompt, user_query, context, memory):
    # Format the prompt with the user query, context, and history
    formatted_prompt = final_prompt.format_messages(
        question=user_query, 
        context=context, 
        chat_history=memory.chat_memory.messages  # Correctly access the conversation history
    )
    
    # Print the final prompt for debugging purposes
    print("\n--- FINAL PROMPT SENT TO LLM ---")
    for msg in formatted_prompt:
        # Access the type and content correctly
        print(f"{msg.type.capitalize()} message: {msg.content}")  # Use `type` instead of `role`
    print("\n------------------------------------\n")





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
        doc.metadata["source"] = "Fall Course Registration"
        doc.metadata["category"]="Course Registration"
        doc.metadata["semester"]="Fall 2025"
        doc.page_content = "This is related to courses available to register for fall: " + doc.page_content
        course_docs.append(doc)

    return course_docs  # return a flat list of Document objects   ***********degree_docs +



def enrich_chunks_with_metadata(chunks, llm):
    summarizer_prompt = PromptTemplate.from_template(
        "You are a document organizer.\n"
        "Summarize the following document chunk in 1-2 sentences:\n\n{chunk}"
    )
    tagger_prompt = PromptTemplate.from_template(
        "You are a classifier. Classify the content into one of the categories: "
        "['Degree Requirements', 'Course Info', 'Policies', 'Registration'].\n\n"
        "Chunk:\n{chunk}"
    )
    question_prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Generate 3 relevant and concise student questions based on the following academic content:\n\n{chunk}"
    )

    enriched_chunks = []

    for chunk in chunks:
        text = chunk.page_content

        # Format the prompt to a string
        summary_prompt_str = summarizer_prompt.format_prompt(chunk=text).to_string()
        tag_prompt_str = tagger_prompt.format_prompt(chunk=text).to_string()
        question_prompt_str = question_prompt.format_prompt(chunk=text).to_string()

        # Invoke the LLM with prompt string (or you can pass list of messages)
        summary = llm.invoke(summary_prompt_str).content.strip()
        tag = llm.invoke(tag_prompt_str).content.strip()
        questions_raw = llm.invoke(question_prompt_str).content.strip()

        clean_questions = [q.lstrip("- ").strip() for q in questions_raw.split("\n") if q.strip()]

        chunk.metadata["summary"] = summary
        chunk.metadata["tag"] = tag
        chunk.metadata["sample_questions"] = "\n".join(clean_questions)

        enriched_chunks.append(chunk)
    return enriched_chunks




def webBaseLoader():
    urls=["https://catalog.lamar.edu/college-arts-sciences/computer-science/computer-science-ms/#text",
          "https://www.lamar.edu/financial-aid/withdrawing-and-the-60-dates.html",
        # "https://www.lamar.edu/arts-sciences/computer-science/degrees/graduate/degree-requirements.html",
         "https://catalog.lamar.edu/college-arts-sciences/computer-science/computer-science-ms/#requiredcoursestext",
         ]
    loader=UnstructuredURLLoader(urls)
    docs= loader.load()
    return docs

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


load_dotenv()

clarify_prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Rewrite this question for clarity and completeness: {question}"
)


system_msg = SystemMessagePromptTemplate.from_template(
    "You are an academic advisor for Computer Science graduate students.\n\n"
    "You must:\n"
    "1. Recommend only courses explicitly offered in Fall 2025 for the registration (as shown in the documents).\n"
    "2. Recommend courses for registration, only if the {question} has satisfied the listed prerequisites courses. if not simply say, no\n"
    "3. In the {chat_history} not done with prerequisite course don't suggest the course in the response for registration, and say no if {question} ask for the course"
    "3. Do not recommend courses for registration, in the {chat_history} has already completed.\n"
    "4. Use only the documents provided (no outside knowledge or assumptions).\n"
    "5. Consider the courses as completed, if and only mentioned in the {chat_history} (Do not assume any course that is completed unless mentioned in the {chat_history})"
    "6. In the {chat_history} you see like the courses met with the pre-requisites, you can suggest, "
    "If a course does not meet the above criteria, do not include it.\n"
)


# system_msg = SystemMessagePromptTemplate.from_template(
#     "You are an expert Academic Advisor for Computer Science Graduate Students, Who helps in student course registration." \
#     " Only answer based on the given documents. Do not guess."
# )

human_msg = HumanMessagePromptTemplate.from_template(
    "Use the following documents to answer the questions {context}\n\n Student Qusetion: {question}\n\n use the following chat history if required {chat_history}"

    # "Student Question: {question}\n\n Relevent Context:\n{context} \n Conversation History : {chat_history}"
)

final_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# --------------------------------------------------------------------------------------------------------------------------------


# folder_path = "Rag_CS_Documents"
# docs= document_parsing(folder_path=folder_path)


# docs= webBaseLoader()
# docs.extend(degree_audit())
docs=degree_audit()


splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
chunks = splitter.split_documents(docs)





metadata_llm=ChatOllama(model="mistral-small",temperature=0.4)
# metadata_llm=ChatOpenAI(model_name="gpt-4",temperature=0.4)

# chunks_with_metadata= enrich_chunks_with_metadata(chunks,metadata_llm)
    



embeddings = OllamaEmbeddings(model="mxbai-embed-large")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
persist_directory="./rag_data"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    #  Load existing vector DB
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    #  Create it for the first time
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory) #Remember chunks - updated Chunks
    vector_store.persist()


retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 12})

llm=ChatOllama(model="qwen2.5",temperature=0.3)
# llm = ChatOpenAI(model_name="gpt-4",temperature=0.1)

clarify_chain = clarify_prompt | llm

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
    # print_prompt(final_prompt, query, result["source_documents"], memory)
    print(f"\nBot: {result['answer']}\n")


# retrieved_docs = retriever.get_relevant_documents("what are the courses available")
# # Print the retrieved chunks
# for i, doc in enumerate(retrieved_docs):
#     print(f"\n--- Retrieved Chunk {i+1} ---")
#     print(doc.metadata)
#     print("-------------------metedata----------------------")
#     print(doc.page_content)
#     print("-----------------------------------------------------------------------------------------------------------------")
