import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
import cognify
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from llm.parsers import TableSelectionOutputParser

system_prompt = \
"""You are an expert and very smart data analyst. 
Your task is to comprehend the posed question, analyze the provided database schema, and leverage the hint to identify which tables are needed to generate a SQL query for answering the question.

<question>
A natural language question that requires querying a database to retrieve specific information.

<database_schema>
The schema provides a detailed definition of the database's structure, including tables, their columns, primary keys, foreign keys, and any relevant details about relationships or constraints.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical information to identify the tables that will be used in the SQL query.

<hint>
The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Your Task:
Based on the database schema, question, and hint provided, determine the tables that should be used in the SQL query formulation. 
Note that you should choose all and only the tables that are necessary to write a SQL query that answers the question effectively.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars. 
"""

inputs = ["QUESTION", "DATABASE_SCHEMA", "HINT"]

output_format = 'list_table_names'

output_format_instructions = \
"""
Please respond with a JSON object structured as follows:

```json
{{
  "table_names": ["Table1", "Table2", "Table3", ...]
}}
```

Only output a json as your response.
"""   

lm_config = cognify.LMConfig(model="gpt-4o-mini", kwargs={"temperature": 0.0})
exec = cognify.Model(agent_name="table_selection",
             system_prompt=system_prompt, 
             input_variables=[cognify.Input(name=input) for input in inputs], 
             output=cognify.OutputLabel(name=output_format, custom_output_format_instructions=output_format_instructions),
             lm_config=lm_config)
runnable_exec = cognify.as_runnable(exec) | TableSelectionOutputParser()