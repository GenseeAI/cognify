<p align="center">
    <img src="https://github.com/GenseeAI/cognify/blob/main/cognify.jpg?raw=true" alt="Cognify logo">
</p>

<p align="center">
| <a href="https://cognify-ai.readthedocs.io/en/latest/user_guide/quickstart.html"><b>Quickstart</b></a> | <a href="https://cognify-ai.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2502.08056"><b>Research Paper</b></a> | <a href="https://www.gensee.ai/blog"><b>Blog</b></a> | <a href="https://discord.gg/8TSFeZA3V6"><b>Discord</b></a> | <a href="https://forms.gle/Be3MD3pGPpZaUmrVA"><b>Send Feedback</b></a> | 
</p>

# Multi-Faceted Gen-AI Workflow Autotuning

Building high-quality, cost-effective generative-AI (gen-AI) applications is challenging due to the absence of systematic methods for tuning, testing, and optimizing them. 
We introduce **Cognify**, a tool that automatically enhances generation quality, decreases execution latency, and reduces execution monetary costs for gen-AI workflows, including those written with LangChain, LangGraph, DSPy, and annotated Python. 
Built on a novel foundation of hierarchical, workflow-level optimization, Cognify improves workflow **generation quality by up to 2.8x**, reduces execution **cost
by up to 10x**, and reduces end-to-end **latency by up to 2.7x** compared to original expert-written workflows. 

## News

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
cognify optimize /your/gen/ai/workflow.py   
```
where `workflow.py` is your workflow source code. Cognify currently supports unmodified [LangChain](https://github.com/langchain-ai/langchain) and [DSPy](https://github.com/stanfordnlp/dspy) workflow source code. You can also port your existing workflow written directly on Python or develop new Python-based workflows with our [simple workflow interface](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/interface/program.html).

Cognify automatically searches for a `config.py` in the same folder as the workflow. You can also specify this file explicitly by:
```bash
cognify optimize /your/gen/ai/workflow.py -c /your/gen/ai/custom_config.py
```

Within the `config.py`, you should define the following:

- **Sample Dataset**: Cognify relies on training data to evaluate and improve its workflow optimization. You should provide a data loader that loads your training dataset in the form of input-output pairs. Read more about how to [load your data](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/dataloader.html).
- **Evaluator**: Cognify expects you to provide an evaluator for judging the final workflow generation's quality. To help you get started, Cognify provides several common evaluator implementations such as the F1 score. Find out more about [workflow evaluator](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/evaluator.html).
- **Optimization Configurations and Model Set Selection**: Optionally, you can configure your optimization process in various ways. For example, you can choose between light, medium, or heavy search over the optimization space. We also provide a few domain-specific optimization configurations. You can also define the set of models you want Cognify to explore. If no configurations or models are provided, Cognify uses a default set of values. Read more about [optimization configurations and model selections](https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/optimizer.html).

With these parameters, Cognify optimizes your workflow by iteratively experimenting with various combinations of tuning methods (we call them “*cogs*”) applied across workflow components and assessing the effectiveness of these combinations based on the quality of the final output using the user-provided sample dataset and evaluator. This process continues until Cognify hits the user-specified maximum iteration numbers (in `config.py`).

The result of this process is a set of optimized workflow versions with different quality-cost-latency combinations on the [Pareto frontier](https://en.wikipedia.org/wiki/Pareto_front) among all the iterations.
You can inspect the optimizations applied in these output files 
You can evaluate these optimized versions with a test dataset:

```bash
cognify evaluate /your/gen/ai/workflow/optimization/output/path
```

You can also continue running Cognify with more iterations from these outputs using the `-r` or `--resume` flag:

```bash
cognify optimize /your/gen/ai/workflow.py -r
```

Follow our [quickstart](https://cognify-ai.readthedocs.io/en/latest/user_guide/quickstart.html) or read our full [documentation](https://cognify-ai.readthedocs.io/en/latest/) to learn more.

- [User Guide](https://cognify-ai.readthedocs.io/en/latest/user_guide/): A Cognify user guide consists of a quickstart and a step-by-step tutorial
- [Fundamentals](https://cognify-ai.readthedocs.io/en/latest/fundamentals/): Fundamental concepts about Cognify internals
- [API Reference](https://cognify-ai.readthedocs.io/en/latest/api_ref/modules.html)

Tell us [how you use Cognify](https://forms.gle/Be3MD3pGPpZaUmrVA)!
