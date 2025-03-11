import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
import cognify
from langchain_core.output_parsers import JsonOutputParser
from llm.parsers import ColumnSelectionOutput

system_prompt = \
"""You are an expert and very smart data analyst.
Your task is to understand the posed question, examine the provided database schema, and use the hint to pinpoint the specific columns within tables that are essential for crafting a SQL query to answer the question.

<question>
A natural language question that requires querying a database to retrieve specific information.

<database_schema>
The schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical information to identify the columns that will be used in the SQL query.

<hint>
The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Your Task:
Based on the database schema, question, and hint provided, identify all and only the columns that are essential for crafting a SQL query to answer the question.
Tip: If you are choosing a column for filtering a value within that column, make sure that column has the value as an example.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.
"""

inputs = ["QUESTION", "DATABASE_SCHEMA", "HINT"]

output_format = 'selected_columns'

output_format_instructions = \
"""Please respond with a JSON object structured as follows:

```json
{{
  "table_name1": ["column1", "column2", ...],
  "table_name2": ["column1", "column2", ...],
  ...
}}
```

Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the question. Only output a json as your response.
"""
lm_config = cognify.LMConfig(model="gpt-4o-mini", kwargs={"temperature": 0.0})
exec = cognify.Model(agent_name="column_selection",
             system_prompt=system_prompt, 
             input_variables=[cognify.Input(name=input) for input in inputs], 
             output=cognify.OutputLabel(name=output_format, custom_output_format_instructions=output_format_instructions),
             lm_config=lm_config)
runnable_exec = cognify.as_runnable(exec) | JsonOutputParser()