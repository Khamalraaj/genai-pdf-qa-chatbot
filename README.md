## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

    Parsing and indexing the content of a PDF document.
    Allowing users to ask questions in natural language.
    Providing concise and accurate answers based on the content of the document.

The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

#### STEP 1: 
Load and Parse PDF
Use LangChain's DocumentLoader to extract text from a PDF document.

#### STEP 2: 
Create a Vector Store
Convert the text into vector embeddings using a language model, enabling semantic search.

#### STEP 3: 
Initialize the LangChain QA Pipeline
Use LangChain's RetrievalQA to connect the vector store with a language model for answering questions.

#### STEP 4: 
Handle User Queries
Process user queries, retrieve relevant document sections, and generate responses.

#### STEP 5: 
Evaluate Effectiveness
Test the chatbot with a variety of queries to assess accuracy and reliability.

### PROGRAM:
```py
# Cell 1: Install required libraries (run once)
!pip install openai langchain langchain-openai langchain-community langchain-chroma panel param pypdf docarray
# Cell 2: Set your OpenAI API key directly
import os
os.environ["OPENAI_API_KEY"] = "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"  # <-- Replace with your actual key
print("API key set.")
!pip install langchain-text-splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   # <-- fix this line

loader = PyPDFLoader("AI_and_Data_Science.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = splitter.split_documents(documents)
!pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="docs/chroma/"
)
print("DB created and saved.")
# Cell 4: Test similarity searc h
question = "What is the difference between Narrow AI and General AI?"
docs = vectordb.similarity_search(question, k=3)
print(f"Found {len(docs)} relevant documents.")
!pip install langchain-groq
from langchain_groq import ChatGroq

import os
os.environ["GROQ_API_KEY"] = "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"  # paste your key here

llm_name = "llama-3.1-8b-instant"  # updated model name
llm = ChatGroq(model_name=llm_name, temperature=0)
print("Name: kHAMALRAAJ s")
response = llm.invoke("Hello world!")
print(response.content)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Use three sentences maximum. Keep the answer concise.
Always say "thanks for asking!" at the end.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": vectordb.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | QA_CHAIN_PROMPT
    | llm
    | StrOutputParser()
)

question = "What are the ethical concerns in AI?"
result = qa_chain.invoke(question)
print(result)



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below. Keep it concise.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa = (
    {
        "context": vectordb.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: chat_history
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = "Who coined the term Machine Learning?"
answer = qa.invoke(question)
chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
print(answer)
```

### OUTPUT:
<img width="1092" height="59" alt="image" src="https://github.com/user-attachments/assets/fff86d88-824b-4370-aaa5-cec2c63e64b3" />
<img width="1413" height="201" alt="image" src="https://github.com/user-attachments/assets/bcf8d8b2-86a4-42ed-8c3f-d901402c77d6" />
<img width="991" height="198" alt="image" src="https://github.com/user-attachments/assets/659b0de7-2ab1-49e0-b9c4-0fd3aaf08764" />


### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
