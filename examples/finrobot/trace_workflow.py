import re
from collections import defaultdict

import dspy.teleprompt
from prompts import leader_system_message, role_system_message
import json
from pydantic import BaseModel
import dotenv
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from dspy_agents import *
from config import load_all_data, evaluate_for_dspy_trace
import time
from tqdm import tqdm
from opto.trace import node, bundle, model, ExecutionError
from opto.trace.nodes import GRAPH, ParameterNode
from opto.optimizers import OptoPrime
import autogen
import litellm
from trace_agents import *

dotenv.load_dotenv()

def construct_fin_agent(agent_name, agent_role):
    if agent_name == "Value_Factor_Researcher":
        return ValueFactorResearcher(agent_role)
    elif agent_name == "Growth_Factor_Researcher":
        return GrowthFactorResearcher(agent_role)
    elif agent_name == "Momentum_Factor_Researcher":
        return MomentumFactorResearcher(agent_role)
    elif agent_name == "Quality_Factor_Researcher":
        return QualityFactorResearcher(agent_role)
    elif agent_name == "Volatility_Factor_Researcher":
        return VolatilityFactorResearcher(agent_role)
    elif agent_name == "Liquidity_Factor_Researcher":
        return LiquidityFactorResearcher(agent_role)
    elif agent_name == "Sentiment_Factor_Researcher":
        return SentimentFactorResearcher(agent_role)
    elif agent_name == "Macro_Factor_Researcher":
        return MacroFactorResearcher(agent_role)
    elif agent_name == "Portfolio_Manager":
        return PortfolioManager(agent_role)
    elif agent_name == "Quantitative_Analyst":
        return QuantitativeAnalyst(agent_role)
    elif agent_name == "Financial_Data_Specialist":
        return FinancialDataSpecialist(agent_role)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")
    
@bundle(trainable=True, allow_external_dependencies=True)
def parse_order_string(order_string: str):
    """Parse the order string, which is expected to follow '[<name of staff>] <order>'."""
    pattern = r"\[(.*?)\]\s+(.*)"
    match = re.search(pattern, order_string)
    
    if match:
        name = match.group(1)  # Extract name inside square brackets
        order = match.group(2)  # Extract the order instruction
        return name, order
    else:
        raise ValueError(f"Invalid order string format: {order_string}. Ensure it follows '[<name of staff>] <order>'.")

with open('agent_profiles.json') as f:
    profiles = json.load(f)
group_members: dict[str, FinAgent] = {}
for profile in profiles:
    agent_name = profile['name']
    agent_role = profile['profile']
    group_members[agent_name] = construct_fin_agent(agent_name, agent_role)


class LeaderCallOutput(BaseModel):
    project_status: str
    member_order: str
    solution: str

@model
class FinRobot():
    def __init__(self, k=5):
        self.k = k
        self.task_history = defaultdict(list)
        self.prompt = ParameterNode(leader_prompt, trainable=True, description="[ParameterNode] This is the prompt template for the group leader's LLM call.")

    def call_group_leader(self, task, project_history, remaining_order_budget):
        messages = [
            {"role": "system", "content": self.prompt.data},
            {"role": "user", "content": f"Given the following task, project history, and remaining order budget, please provide the project status, upcoming member order, and the solution at this stage.\n\nTask: {task} \n\nProject History: {project_history} \n\nRemaining Order Budget: {remaining_order_budget}"},
        ]
        response = litellm.completion("gpt-4o-mini", messages, response_format=LeaderCallOutput)
        return LeaderCallOutput(**json.loads(response.choices[0].message.content))
        

    @bundle(trainable=True, allow_external_dependencies=True)
    def forward(self, task):
        """Select agents to solve the task and collect responses."""
        for i in range(self.k):
            project_hist = format_history(self.task_history[task])
            leader_msg = self.call_group_leader(
                task=task, project_history=project_hist, remaining_order_budget=self.k - i
            )
            
            if leader_msg.project_status == "END":
                return leader_msg
            
            try:
                result = parse_order_string(leader_msg.member_order).data
                member_name, member_order = result
                if member_name in group_members:
                    member_response = group_members[member_name].solve_order(member_order, task)
                    
                else:
                    member_response = "Invalid group member was chosen."
                self.task_history[task].append(
                    f"Order: {member_name} - {member_order}\n"
                    f"Member Response: {member_response}"
                )
            except ValueError as e:
                self.task_history[task].append(
                    f"Error in parsing the member order"
                )
        else:
            return dspy.Prediction(solution="Project not completed in time.")

        
fin_robot = FinRobot(k=3)

lm = dspy.LM("gpt-4o-mini", temperature=0, cache=False)
dspy.settings.configure(lm=lm)

## load data ##
all_train, _, all_test = load_all_data()

# convert to examples
def convert_to_example(input):
    return dspy.Example({"task": input[0]["task"], "mode": input[0]["mode"], "label": input[1]["label"]}).with_inputs("task")

all_train_examples = [convert_to_example(input) for input in all_train]
all_test_examples = [convert_to_example(input) for input in all_test]

## train
epochs = 1
total_train_acc = 0

optimizer = OptoPrime(fin_robot.parameters(), config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

for i in range(epochs):
    for j, example in enumerate(tqdm(all_train_examples)):
        GRAPH.clear()
        try:
            response = fin_robot(task=example.task)
            try:
                correctness = evaluate_for_dspy_trace(example, response.data)
                if correctness > 0.8:
                    feedback = "The answer is correct! No need to change anything."
                else:
                    feedback = f"The answer is wrong. We expect the output of your answer to be \"{example.label}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."
                no_error = True
                total_train_acc += correctness
            except:
                correctness = 0
                no_error = False
        except ExecutionError as e:
            response = e.exception_node
            feedback = response.data
            correctness = 0
            no_error = False

        optimizer.zero_feedback()
        optimizer.backward(response, feedback)

        print(f"output={response.data}, feedback={feedback}\n")  # logging

        optimizer.step(verbose=False)
        checkpoint_name = f"trace_results/epoch_{i}/{j}.pkl"
        fin_robot.save(checkpoint_name)

## evaluation
def eval(test_set):
    total_score = 0
    for example in tqdm(test_set):
        try:
            result = fin_robot(task=example.task)
            score = evaluate_for_dspy_trace(example, result.data)
            total_score += score
        except:
            pass
    return total_score / len(test_set)

val_acc = eval(all_test_examples)
