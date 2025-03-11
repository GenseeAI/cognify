import litellm
from functools import partial
from pydantic import BaseModel, Field

class Judgement(BaseModel):
    score: float = Field(description="The score given by the judge")

def llm_judge_generic(llm_output, ground_truth, model='gpt-4o'):
    system_prompt = """
        You are an expert evaluator assessing whether the generated output by the model effectively
        expresses the intended meaning of the ground-truth output on a scale of 1 to 5.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generated output:\n\n{llm_output}\n\nGround-truth output:\n\n{ground_truth}"},
    ]
    
    response = litellm.completion(model,
                                  messages,
                                  response_format=Judgement,
                                  temperature=0.0)
    judgement: Judgement = Judgement.model_validate_json(response.choices[0].message.content)
    return judgement.score

def llm_judge_factory(model='gpt-4o'):
    return partial(llm_judge_generic, model=model)