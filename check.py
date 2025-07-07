from langchain_community.llms import Ollama
from config import raw_documents
from document_loader import Vectorstore
from langchain_core.prompts import PromptTemplate
llm = Ollama(model="gemma2")
documents = Vectorstore(raw_documents)
from langchain_core.runnables import Runnable

def ask_with_vectorstore(message: str, vectorstore, llm, chat_history=[]):
    # Step 1: Retrieve relevant chunks
    docs = vectorstore.retrieve(message)  # returns list of dicts with 'text'

    # Step 2: Format document context
    context = "\n\n".join([doc["text"] for doc in docs])

    # Step 3: Format history
    history_str = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history])

    # Step 4: Create prompt
    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant for our website Autobg.ai answering questions for people who wants help with out product. Use the following context to answer the user's question in short.

            Context:
            {context}
            
            {history}
            User: {question}
            Assistant:"""
    )

    # Step 5: Pipe prompt into LLM
    chain: Runnable = prompt | llm

    # Step 6: Invoke chain
    response = chain.invoke({
        "context": context,
        "question": message,
        "history": history_str
    })

    print("Chatbot:", response)

    # Step 7: Update history
    chat_history.append({"user": message, "assistant": response})

    return chat_history

chat_history = []
while True:
    user_input = input("Query:")
    chat_history  = ask_with_vectorstore(message=user_input, vectorstore=documents, llm=llm, chat_history=chat_history)
    print(chat_history)