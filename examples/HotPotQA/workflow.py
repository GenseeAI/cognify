import os
import dotenv
dotenv.load_dotenv()

import dspy

gpt4o = dspy.LM('gpt-4o', max_tokens=1000)
colbert = dspy.ColBERTv2(url=os.environ['COLBERT_URL'])

dspy.configure(lm=gpt4o, rm=colbert)

from dsp.utils.utils import deduplicate

class Question2Query(dspy.Signature):
    """
    You are an expert at crafting precise search queries based on a provided question. Your task is to generate a well-structured search query that will help retrieve relevant external documents containing information needed to answer the question.
    """
    question: str = dspy.InputField()
    search_query: str = dspy.OutputField()
    
class ContextQuestion2Query(dspy.Signature):
    """
    You are good at extract relevant details from the provided context and question. Your task is to propose an effective search query that will help retrieve additional information to answer the question. The search query should target the missing information while avoiding redundancy.
    """
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    search_query: str = dspy.OutputField()
    
class ContextQuestion2Answer(dspy.Signature):
    """
    You are an expert at answering questions based on provided documents. Your task is to formulate a clear, accurate, and concise answer to the given question by using the retrieved context (documents) as your source of information. Please ensure that your answer is well-grounded in the context and directly addresses the question.
    """
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query_0 = dspy.Predict(Question2Query)
        self.generate_query_1 = dspy.Predict(ContextQuestion2Query)
        self.generate_answer = dspy.Predict(ContextQuestion2Answer)

    def forward(self, question):
        context = []

        search_query = self.generate_query_0(question=question).search_query
        passages = self.retrieve(search_query).passages
        context = deduplicate(context + passages)
        
        search_query = self.generate_query_1(context=context, question=question).search_query
        passages = self.retrieve(search_query).passages
        context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question).answer
        return answer
    
agent = BasicMH(passages_per_hop=2)

import cognify

@cognify.register_workflow
def qa_workflow(question):
    answer = agent(question=question)
    return {'answer': answer}

if __name__ == "__main__":
    print(qa_workflow(question="What was the 2010 population of the birthplace of Gerard Piel?"))