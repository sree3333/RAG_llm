from langchain_community.llms import Ollama
from document_loader import Vectorstore
from config import raw_documents
# from processing_chat import ask_with_vectorstore
from Processing_chat_RAG import ask_with_rag
llm = Ollama(model="gemma3:1b")
documents = Vectorstore(raw_documents)

from langchain_core.prompts import PromptTemplate

def run_chatbot(message: str, chat_history: list = []):

    # try:
    chat_history = ask_with_rag(message=message, vectorstore=documents, llm=llm, chat_history=chat_history)
    # except:
    #     chat_history = []
    return chat_history



chat_history = []
while True:
    user_input = input("Query:")
    chat_history  = run_chatbot(message=user_input, chat_history=chat_history)
    # print(chat_history)
    print('Assistant:',chat_history[-1]['assistant'])
