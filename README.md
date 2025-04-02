<p align="center">
    <img src="https://github.com/GenseeAI/cognify/blob/main/cognify.jpg?raw=true" alt="Cognify logo">
</p>

<p align="center">
| <a href="https://cognify-ai.readthedocs.io/en/latest/user_guide/quickstart.html"><b>Quickstart</b></a> | <a href="https://cognify-ai.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2502.08056"><b>Research Paper</b></a> | <a href="https://www.gensee.ai/blog"><b>Blog</b></a> | <a href="https://discord.gg/8TSFeZA3V6"><b>Discord</b></a> | <a href="https://forms.gle/Be3MD3pGPpZaUmrVA"><b>Send Feedback</b></a> | 
</p>

# Multi-Faceted AI Agent and Workflow Optimizer

Building high-quality, cost-effective gen-AI applications such as AI agents is challenging due to the absence of systematic methods for tuning, testing, and optimizing them. 
We introduce **Cognify**, a tool that automatically enhances generation quality, decreases execution latency, and reduces execution monetary costs for AI agents and workflows, including those written with LangChain, LangGraph, DSPy, and Cognify's own framework. 
Built on a novel foundation of hierarchical, workflow-level optimization, Cognify improves agent **generation quality by up to 2.8x**, reduces execution **cost
by up to 10x**, and reduces end-to-end **latency by up to 2.7x** compared to original expert-written agents. 

## News

- <b>Mar 13, 2025</b>: New blog series (Pt2) deep-diving into how Cognify achieves 2.8x quality improvement with just $5 and 24 minutes available [here](https://www.gensee.ai/post/auto-tuning-with-cognify-the-secret-to-boosting-your-gen-ai-workflow-quality-by-2-8-times-with-5-i-1)
- <b>Mar 12, 2025</b>: New blog series (Pt2) introducing how Cognify achieves 2.8x quality improvement with just $5 and 24 minutes available [here](https://www.gensee.ai/post/auto-tuning-with-cognify-the-secret-to-boosting-your-gen-ai-workflow-quality-by-2-8-times-with-5-i)
- <b>Feb 12, 2025</b>: Cognify [research paper](https://arxiv.org/abs/2502.08056) available on arxiv.
- <b>Nov 25, 2024</b>: Cognify announcement and overview blog post published [here](https://mlsys.wuklab.io/posts/cognify/).

## Installation

Cognify is available as a Python package and can be installed as
```
pip install cognify-ai
```

Or install from the source:
```
git clone https://github.com/GenseeAI/cognify
cd cognify
pip install -e .
```

## Getting Started

You can use Cognify with our simple CLI:
```bash
cognify optimize /your/ai/agent.py   
```
where `agent.py` is your agent source code. Cognify currently supports unmodified [LangChain](https://github.com/langchain-ai/langchain) and [DSPy](https://github.com/stanfordnlp/dspy) source code. You can also port your existing agent written directly on Python or develop new Python-based workflows with our [simple workflow framework](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/interface/program.html).

Cognify automatically searches for a `config.py` in the same folder as the agent. You can also specify this file explicitly by:
```bash
cognify optimize /your/ai/agent.py -c /your/ai/custom_config.py
```

Within the `config.py`, you should define the following:

- **Sample Dataset**: Cognify relies on training data to evaluate and improve its agent optimization. You should provide a data loader that loads your training dataset in the form of input-output pairs. Read more about how to [load your data](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/dataloader.html).
- **Evaluator**: Cognify expects you to provide an evaluator for judging the final agent generation's quality. To help you get started, Cognify provides several common evaluator implementations such as the F1 score. Find out more about [workflow evaluator](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/evaluator.html).
- **Optimization Configurations and Model Set Selection**: Optionally, you can configure your optimization process in various ways. For example, you can choose between light, medium, or heavy search over the optimization space. We also provide a few domain-specific optimization configurations. You can also define the set of models you want Cognify to explore. If no configurations or models are provided, Cognify uses a default set of values. Read more about [optimization configurations and model selections](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/optimizer.html).

With these parameters, Cognify optimizes your agent by iteratively experimenting with various combinations of tuning methods (we call them “*cogs*”) applied across agent (workflow) components and assessing the effectiveness of these combinations based on the quality of the final output using the user-provided sample dataset and evaluator. This process continues until Cognify hits the user-specified maximum iteration numbers (in `config.py`).

The result of this process is a set of optimized agent versions with different quality-cost-latency combinations on the [Pareto frontier](https://en.wikipedia.org/wiki/Pareto_front) among all the iterations.
You can inspect the optimizations applied in these output files 
You can evaluate these optimized versions with a test dataset:

```bash
cognify evaluate /your/ai/agent/optimization/output/path
```

You can also continue running Cognify with more iterations from these outputs using the `-r` or `--resume` flag:

```bash
cognify optimize /your/ai/agent.py -r
```

Follow our [quickstart](https://cognify-ai.readthedocs.io/en/latest/user_guide/quickstart.html) or read our full [documentation](https://cognify-ai.readthedocs.io/en/latest/) to learn more.

- [User Guide](https://cognify-ai.readthedocs.io/en/latest/user_guide/): A Cognify user guide consists of a quickstart and a step-by-step tutorial
- [Fundamentals](https://cognify-ai.readthedocs.io/en/latest/fundamentals/): Fundamental concepts about Cognify internals
- [API Reference](https://cognify-ai.readthedocs.io/en/latest/api_ref/modules.html)

Tell us [how you use Cognify](https://forms.gle/Be3MD3pGPpZaUmrVA)!
