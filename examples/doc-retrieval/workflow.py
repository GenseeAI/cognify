import os
import dotenv
dotenv.load_dotenv()

import dspy

gpt4o_mini = dspy.LM('gpt-4o-mini', max_tokens=1000)
colbert = dspy.ColBERTv2(url=os.environ['COLBERT_URL'])

dspy.configure(lm=gpt4o_mini, rm=colbert)


class Summarize1(dspy.Signature):
    """
    You are a research analyst specializing in fact-checking complex claims through evidence gathering. Your task is to examine the provided passages carefully and summarize key points that relate directly to the claim. Highlight essential details, contextual insights, and any information that may either support or provide background on the claim. Your summary should focus on clarity and relevance, setting the stage for deeper investigation.
    """
    claim: str = dspy.InputField()
    passages: str = dspy.InputField()
    summary: str = dspy.OutputField()

class CreateQueryHop2(dspy.Signature):
    """
    You are a strategic search specialist skilled in crafting precise queries to uncover additional evidence. Your task is to generate a focused and clear query that will help retrieve more relevant external documents. This query should aim to address gaps, ambiguities, or details missing in the existing information. Target specific information or clarifications that could strengthen the evidence for or against the claim.
    """
    claim: str = dspy.InputField()
    summary: str = dspy.InputField()
    query: str = dspy.OutputField()
    
class Summarize2(dspy.Signature):
    """
    You are an evidence synthesis expert specializing in extracting distinct, complementary insights to deepen understanding of claims. Your task is to summarize the new passages, emphasizing any new details that support, refute, or add depth to the claim. Your summary should provide unique and complementary knowledge base that is not covered in the provided context to advance the understanding of the claim.
    """
    claim: str = dspy.InputField()
    context: str = dspy.InputField()
    passages: str = dspy.InputField()
    summary: str = dspy.OutputField()
    
class CreateQueryHop3(dspy.Signature):
    """
    You are a strategic search specialist skilled in crafting precise queries to uncover additional evidence. Your task is to generate a focused and clear query that will help retrieve more relevant external documents. This query should aim to address gaps, ambiguities, or details missing in the existing information. Target specific information or clarifications that could strengthen the evidence for or against the claim.
    """
    claim: str = dspy.InputField()
    summary1: str = dspy.InputField()
    summary2: str = dspy.InputField()
    query: str = dspy.OutputField()
    
class RetrieveMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        
        # DSPy retrieval does not return metadata currently
        # We patch this in _retrieve.py
        from _retrieve import _Retrieve
        self.retrieve_k = _Retrieve(k=self.k)
        
        self.create_query_hop2 = dspy.Predict(CreateQueryHop2)
        self.create_query_hop3 = dspy.Predict(CreateQueryHop3)
        self.summarize1 = dspy.Predict(Summarize1)
        self.summarize2 = dspy.Predict(Summarize2)
    
    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim, with_metadata=True)
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs.passages).summary
        
        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query, with_metadata=True)
        summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs.passages).summary
        
        # HOP 3
        hop3_query = self.create_query_hop3(claim=claim, summary1=summary_1, summary2=summary_2).query
        hop3_docs = self.retrieve_k(hop3_query, with_metadata=True)
        
        # get top-10 passages
        scores, pids, passages = [], [], []
        for retrieval in [hop1_docs, hop2_docs, hop3_docs]:
            for score, pid, passage in zip(retrieval.score, retrieval.pid, retrieval.passages):
                scores.append(score)
                passages.append(passage)
                pids.append(pid)

        sorted_passages = sorted(zip(scores, pids, passages), key=lambda x: x[0], reverse=True)[:10]
        scores, pids, passages = zip(*sorted_passages)
        return dspy.Prediction(scores=scores, pids=pids, passages=passages)
    
agent = RetrieveMultiHop()

import cognify

@cognify.register_workflow
def hover_workflow(claim):
    result = agent(claim=claim)
    return {'pred_docs': result.pids}

if __name__ == "__main__":
    claim = "Skagen Painter Peder Severin Kr\u00f8yer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgstr\u00f6m studied with in the early 1900s."
    pred_docs = hover_workflow(claim)
    print(pred_docs)