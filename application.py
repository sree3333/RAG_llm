from langchain_community.llms import Ollama
from document_loader import Vectorstore
from config import raw_documents
llm = Ollama(model="tinyllama")
documents = Vectorstore(raw_documents)

from langchain_core.prompts import PromptTemplate

def run_chatbot(message: str, chat_history: list = []):
    # Simulate search query extraction via prompt
    search_prompt = PromptTemplate.from_template(
        "Given the following user message and chat history, extract concise search queries "
        "that can be used to retrieve relevant documents. Only return the queries as a list.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Message:\n{message}\n\n"
        "Search Queries:"
    )
    search_query = llm.invoke(search_prompt.format(message=message)).strip()

    # documents = documents.

    if search_query:
        print("Retrieving information...", end="")
        # documents = vectorstore.retrieve(search_query)

    # Combine retrieved documents into context
    context_text = "\n\n".join(
        f"[{doc['title']}] {doc['text']}" for doc in documents
    )

    # Build full prompt with history
    history_prompt = "\n".join(
        [f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history]
    )

    final_prompt = f"""
        {history_prompt}
        
        Context:
        {context_text}
        
        User: {message}
        Assistant:"""

    print("\nChatbot:")
    response = llm.invoke(final_prompt)
    print(response)

    # Update chat history
    chat_history.append({"user": message, "assistant": response})

    return chat_history


chat_history = run_chatbot(message="Hello, I have a question")
