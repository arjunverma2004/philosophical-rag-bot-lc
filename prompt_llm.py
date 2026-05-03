from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser





def get_brain():
    template = """
    You are a careful reader of philosophical texts.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Answer ONLY using the provided context.
    - Explain the philosophical ideas in clear terms.
    - If the text includes reasoning or arguments, break them down step-by-step.
    - Maintain fidelity to the author's meaning (do not distort or oversimplify).
    - Do NOT add external philosophical knowledge or modern interpretations.
    - If unclear or missing, respond:
      "The text does not provide a clear answer."

    Structure your response as:
    1. Direct answer
    2. Explanation (based on the text)
    3. Key idea summary

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

    parser = StrOutputParser()

    chain = prompt | llm | parser
    
    return chain




if __name__ == "__main__":    

    chain = get_brain()

    response = chain.invoke({
        "context": "In 'The Stranger', Camus explores the absurdity of life...",
        "question": "What is the main philosophical theme?"
    })

    print(response)