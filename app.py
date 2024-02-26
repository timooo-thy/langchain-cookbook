from langchain_core.messages import AIMessage, HumanMessage
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langserve import add_routes
from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langserve.pydantic_v1 import BaseModel
load_dotenv()


llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
vector = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector.as_retriever()

first_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Correct and grammatically complete sentences is required. Only return the query."),
])

retriever_chain = create_history_aware_retriever(llm, retriever, first_prompt)

second_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the following question based only on the provided context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

chat_history = [
    HumanMessage(content="Hello what schools are mentioned?"),
    AIMessage(
        content="The website mentioned in the context is Singapore University of Technology and Design.")
]

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_messages(chat_history)

document_chain = create_stuff_documents_chain(llm, second_prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

final_chain = (
    RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            memory.load_memory_variables) | itemgetter("history")
    )
    | retrieval_chain
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


class Input(BaseModel):
    input: str


class Output(BaseModel):
    answer: str


add_routes(
    app,
    final_chain.with_types(input_type=Input, output_type=Output),
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
