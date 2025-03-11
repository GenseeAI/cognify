import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
import cognify
from llm.parsers import SQLRevisionOutput, RawSqlOutputParser
from langchain_core.runnables import chain

system_prompt = \
"""
You are an expert in SQL query generation and optimization. Your task is to ensure that the SQL query strictly follows the database administrator's instructions and uses the correct conditions. Please revise the given query if it violates any of the instructions or if it needs to be optimized to better answer the question. If the sql query is correct, return the query as it is 

Inputs you will receive:

{{question}}
A natural language question that will use the given query to retrieve specific information for better answering.

{{<sql>}}
The SQL query that needs to be revised

{{database_schema}}
The schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

{{missing_entities}}
A list of entities in the SQL that do not match the database schema.

{{evidence}}
The hint that aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

{{query_result}}
The query result of the given SQL query.

Here is the Database Admin Instructions you must follow when creating or validating a query:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
7. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
8. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
9. Using || ' ' ||  to concatenate is string is banned and using that is punishable by death. Never concatenate columns in the SELECT clause.
10. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
11. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
12. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.


Take a deep breath and think carefully to find the correct sqlite SQL query. If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
"""

inputs = ["QUESTION", "SQL", "DATABASE_SCHEMA", "MISSING_ENTITIES", "EVIDENCE", "QUERY_RESULT"]

output_format = "revised_SQL"

output_format_instructions = \
"""
Please only provide a valid SQL query as your answer. Do not include any additional information or explanations.
"""


lm_config = cognify.LMConfig(model="gpt-4o-mini", kwargs={"temperature": 0.0})
exec = cognify.Model(agent_name="revision",
            system_prompt=system_prompt, 
            input_variables=[cognify.Input(name=input) for input in inputs], 
            output=cognify.OutputLabel(name=output_format, custom_output_format_instructions=output_format_instructions),
            lm_config=lm_config)
raw_runnable_exec = cognify.as_runnable(exec) | RawSqlOutputParser()

@chain
def runnable_exec(input: dict):
    sql = raw_runnable_exec.invoke(input)
    return {"revised_SQL": sql, "chain_of_thought_reasoning": ""}