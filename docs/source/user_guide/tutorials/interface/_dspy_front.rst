.. code-block:: python

    import dspy

    gpt4o = dspy.LM('gpt-4o', max_tokens=300)
    dspy.configure(lm=gpt4o)

    class MathSolverWorkflow(dspy.Module):
        def __init__(self):
            super().__init__()
            self.interpreter_agent = dspy.Predict("problem -> math_model")
            self.solver_agent = dspy.Predict("problem, math_model -> answer")
        
        def forward(self, problem):
            math_model = self.interpreter_agent(problem=problem).math_model
            answer = self.solver_agent(problem=problem, math_model=math_model).answer
            return answer
        
    my_workflow = MathSolverWorkflow()

    import cognify
    
    @cognify.register_workflow
    def math_solver_workflow(workflow_input):
        answer = my_workflow(problem=workflow_input)
        return {"workflow_output": answer}