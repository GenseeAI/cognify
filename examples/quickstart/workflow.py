# ----------------------------------------------------------------------------
# Define a single LLM agent to answer user question with provided documents
# ----------------------------------------------------------------------------

import dotenv
from langchain_openai import ChatOpenAI
# Load the environment variables
dotenv.load_dotenv()
# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define system prompt
system_prompt = """
You are an expert at answering questions based on provided documents. Your task is to provide the answer along with all supporting facts in given documents.
"""

# Define agent routine 
from langchain_core.prompts import ChatPromptTemplate
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: {question} \n\nDocuments: {documents}"),
    ]
)

qa_agent = agent_prompt | model

# Define workflow

def doc_str(docs):
    context = []
    for i, c in enumerate(docs):
        context.append(f"[{i+1}]: {c}")
    return "\n".join(docs)

import cognify

@cognify.register_workflow
def qa_workflow(question, documents):
    format_doc = doc_str(documents)
    answer = qa_agent.invoke({"question": question, "documents": format_doc}).content
    return {'answer': answer}

if __name__ == "__main__":
    question = "What was the 2010 population of the birthplace of Gerard Piel?"
    documents = [
        'Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).',
    ]

    result = qa_workflow(question=question, documents=documents)
    print(result)