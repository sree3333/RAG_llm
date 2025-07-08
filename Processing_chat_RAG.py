from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap
from typing import List, Dict

def ask_with_rag(message: str, vectorstore, llm, chat_history: List[Dict[str, str]] = []):
    # Step 1: Format chat history
    history_str = "\n".join([
        f"User: {entry['user']}\nAssistant: {entry['assistant']}"
        for entry in chat_history
    ])

    # Step 2: Prompt template for final answer generation
    generation_prompt = PromptTemplate.from_template(
        """You are a Polite helpful AI assistant for our website Autobg.ai answering questions related to our product.Never ask for any OTP, card details, or confidential information.


            Context:
            {context}
            
            {history}
            User: {question}
            Assistant:"""
            )

    # Step 3: Define custom retrieval step
    def retrieve_context(inputs: Dict) -> Dict:
        question = inputs["question"]
        docs = vectorstore.retrieve(question)
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(docs)])

        # If no context found, early fallback
        if not docs:
            return {
                "context": "No relevant information was found.",
                "question": question,
                "history": inputs.get("history", "")
            }

        return {
            "context": context,
            "question": question,
            "history": inputs.get("history", "")
        }

    # Step 4: Build RAG chain
    rag_chain: Runnable = (
        RunnableMap({
            "question": lambda _: message,
            "history": lambda _: history_str
        })
        | RunnableLambda(retrieve_context)
        | generation_prompt
        | llm
    )

    # Step 5: Invoke the RAG chain
    response = rag_chain.invoke({})

    # Step 6: Update chat history
    chat_history.append({
        "user": message,
        "assistant": response
    })

    return chat_history
