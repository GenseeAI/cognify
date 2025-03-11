.. Cognify documentation master file, created by
   sphinx-quickstart on Mon Nov  4 09:06:30 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cognify documentation
=====================

.. toctree::
   :maxdepth: 1
   :hidden:

   user_guide/index
   fundamentals/index
   api_ref/modules

**Useful links**:
:ref:`Installation <user_guide_installation>` |
`Source Repository <https://github.com/GenseeAI/cognify>`_ |
`Blog Post <https://mlsys.wuklab.io/posts/cognify/>`_

Cognify is a fully automated optimizer for generative AI (gen-AI) workflows. It transforms user-provided workflows into a set of optimized versions with diverse quality-cost-latency trade-offs. Cognify performs optimizations on individual steps within a workflow and also the workflow structure, with a multi-layer hierarchical optimization framework. Cognify offers improvements of up to 2.8x in generation quality, up to 10x cost reduction, and up to 2.7x reduction in end-to-end latency.



.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: /_static/quickstart.svg
        :text-align: center

        Getting started
        ^^^

        New to Cognify? Check out our hello world example to get started. 
        It shows how to quickly set up Cognify and optimize a simple workflow.

        +++

        .. button-ref:: user_guide/quickstart
            :expand:
            :color: secondary
            :click-parent:

            To the quickstart example

    .. grid-item-card::
        :img-top: /_static/tutorial.svg
        :text-align: center

        Tutorial
        ^^^

        The tutorial explains each step involved in using Cognify. 
        It covers workflow connector, evaluator, dataloader, optimizer configuration, the CLI, and how to interpret the results.

        +++

        .. button-ref:: user_guide/tutorials/overview
            :expand:
            :color: secondary
            :click-parent:

            To the tutorial

    .. grid-item-card::
        :img-top: /_static/fundamentals.svg
        :text-align: center

        Fundamentals
        ^^^

        The fundamentals section provides an in-depth explanation of the key concepts and features of Cognify.

        +++

        .. button-ref:: fundamentals
            :expand:
            :color: secondary
            :click-parent:

            To the building blocks

    .. grid-item-card::
        :img-top: /_static/ref.svg
        :text-align: center

        API Reference
        ^^^
        
        Want to dive into the code? The reference provides detailed information on the functions, classes and data structures used in Cognify.

        +++

        .. button-ref:: api_ref/modules
            :expand:
            :color: secondary
            :click-parent:

            To the API reference