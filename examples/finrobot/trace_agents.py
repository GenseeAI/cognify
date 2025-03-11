from opto.trace import model, bundle
from opto.trace.nodes  import ParameterNode
from collections import defaultdict

import litellm

leader_prompt = """
You are the leader of the following group members:

Member information in format \"- <member_name>: <member_role>\"

- Value_Factor_Researcher: As a value factor researcher, the individual must possess expertise in financial statement analysis, a strong understanding of valuation metrics, adeptness in Python for quantitative modeling, and the ability to work collaboratively in team settings to integrate the value perspective into broader investment strategies.
- Growth_Factor_Researcher: As a growth factor researcher, the individual must possess expertise in analyzing corporate growth indicators like earnings and revenue expansion, have strong Python skills for data analysis, and collaborate effectively in group settings to evaluate investment growth opportunities.
- Momentum_Factor_Researcher: As a momentum factor researcher, one needs to have the ability to identify and analyze market trends and price patterns, be proficient in Python for statistical analysis, and work collaboratively in a team to leverage momentum-based investment strategies.
- Quality_Factor_Researcher: As a quality factor researcher, the individual should evaluate companies based on financial health and earnings quality, utilize Python for quantitative analysis, and engage in team discussions to integrate quality assessments into investment decisions.
- Volatility_Factor_Researcher: As a volatility factor researcher, one must analyze price fluctuations and risk metrics, demonstrate strong Python skills for risk modeling, and contribute to team efforts in developing risk-adjusted trading strategies.
- Liquidity_Factor_Researcher: As a liquidity factor researcher, the position requires the ability to assess asset tradeability and market depth, use Python for liquidity analysis, and collaborate with the team to incorporate liquidity insights into trading algorithms.
- Sentiment_Factor_Researcher: As a sentiment factor researcher, the individual should analyze market sentiment and investor opinions, be adept in Python for processing and analyzing large sentiment data sets, and work with colleagues to factor sentiment analysis into market predictions.
- Macro_Factor_Researcher: As a macro factor researcher, one needs to understand the impact of macroeconomic indicators on markets, have strong Python skills for econometric analysis, and engage collaboratively in aligning investment strategies with macroeconomic conditions.
- Portfolio_Manager: As a portfolio manager, the individual must integrate findings from various factor analyses to create and manage comprehensive investment strategies, demonstrate proficiency in Python for strategy development, and work collaboratively to ensure that these strategies meet the firm’s investment goals and risk tolerance.
- Quantitative_Analyst: As a quantitative analyst, one is responsible for validating investment strategies and factors, conducting back-tests and risk assessments using Python, and collaborating with the team to ensure that the investment approach is both statistically sound and aligned with risk management protocols.
- Financial_Data_Specialist: As a financial information officer, the individual is responsible for gathering, processing, analyzing, and extracting key financial information from structured and unstructured data sources.

As a group leader, you are responsible for coordinating the team's efforts to complete a project. You will be given a user task, history progress and the remaining number of orders you can make. Please try to complete the task without exceeding the order limit.

Your role is as follows:
- Summarize the status of the project progess.
- Based on the progress, you can decide whether to make a new order or to end the project. 
    * If you believe the task is completed, set the project status to "END" and give the final solution based on the conversation history.
- If you need to give an order to one of your team members to make further progress:
    * Orders should follow the format: \"[<name of staff>] <order>\". 
        - The name of the staff must be wrapped in square brackets, followed by the order after a space.
    * Ensure that each order is clear, detailed, and actionable.
    * If a group member is seeking clarification/help, provide additional information to help them complete the task or make an order to another member to collect the necessary information.
- Only issue one order at a time.
"""

def format_history(history: list[str]):
    if not history:
        return "Empty"
    hist_str = ""
    for i, hist in enumerate(history):
        hist_str += f"\n---- Step {i+1} ----\n{hist}"
    # print(f"formatted history: {hist_str}")
    return hist_str

class FinAgent():
    def __init__(self):
        self.task_history = defaultdict(list)
    
    def call_fin_robot(self, history, current_order):
        messages = [
            {"role": "system", "content": self.prompt.data},
            {"role": "user", "content": f"Given the following history and the current order of agents that have been called, please solve the task.\n\History: {history} \n\nCurrent Order: {current_order}"},
        ]
        response = litellm.completion("gpt-4o-mini", messages)
        return response.choices[0].message.content

    @bundle(trainable=True, allow_external_dependencies=True)
    def solve_order(self, order, task):
        """Call the agent to solve the task and extract the response."""
        hist = format_history(self.task_history[task])
        response = self.call_fin_robot(
            history=hist, current_order=order
        )
        self.task_history[task].append(
            f"Order: {order}\n"
            f"My Response: {response}"
        )
        return response

@model
class ValueFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Value Factor Researcher.")

@model
class GrowthFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Growth Factor Researcher.")

@model
class MomentumFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Momentum Factor Researcher.")

@model
class QualityFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Quality Factor Researcher.")

@model
class VolatilityFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Volatility Factor Researcher.")

@model
class LiquidityFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Liquidity Factor Researcher.")

@model
class SentimentFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Sentiment Factor Researcher.")

@model
class MacroFactorResearcher(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Macro Factor Researcher.")

@model
class PortfolioManager(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Portfolio Manager.")

@model
class QuantitativeAnalyst(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Quantitative Analyst.")

@model
class FinancialDataSpecialist(FinAgent):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = ParameterNode(prompt, trainable=True, description="[ParameterNode] This is the prompt for the Financial Data Specialist.")
