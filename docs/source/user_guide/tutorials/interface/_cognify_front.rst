.. code-block:: python

    import cognify

    interpreter_prompt = """
    You are a math problem interpreter. Your task is to analyze the problem, identify key variables, and formulate the appropriate mathematical model or equation needed to solve it. Be concise and clear in your response.
    """
    interpreter_agent = cognify.Model(
        "interpreter", 
        system_prompt=interpreter_prompt, 
        input_variables=[
            cognify.Input("problem")
        ], 
        output=cognify.OutputLabel("math_model"),
        lm_config=cognify.LMConfig(model="gpt-4o", kwargs={"max_tokens": 300})
    )

    solver_prompt = """
    You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
    """
    solver_agent = cognify.Model(
        "solver",
        system_prompt=solver_prompt,
        input_variables=[
            cognify.Input("problem"), 
            cognify.Input("math_model")
        ],
        output=cognify.OutputLabel("answer"),
        lm_config=cognify.LMConfig(model="gpt-4o", kwargs={"max_tokens": 300})
    )

    # Define Workflow
    @cognify.register_workflow
    def math_solver_workflow(workflow_input):
        math_model = interpreter_agent(inputs={"problem": workflow_input})
        answer = solver_agent(inputs={"problem": workflow_input, "math_model": math_model})
        return {"workflow_output": answer}