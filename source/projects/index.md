---
title: Projects
---

# [My GitHub](https://github.com/1337kiwi)

I've been working on a few projects lately. Here are some of them:

## LLM Stuff

I'm working on a mini-curriculum for LLM security, looking at leveraging tools such as [foolbox](https://github.com/bethgelab/foolbox), [cleverhans](https://github.com/cleverhans-lab/cleverhans), and [adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to explore the security properties of LLMs. Also some manual stuff with data poisoning, evasion, and membership inference attacks. It's a WIP at the moment.


[OsGuard](https://github.com/1337kiwi/osguard)

This was done initially as a mini-project for a class, and then became a CTF challenge later on.

This mini project explores the potential of using an LLM as a component of an OS by exploring the capabilities of natural language processing in conversion of human requests into actionable linux shell commands. We start by exploring how an LLM can enforce security properties in CRUD (Create, Read, Update, and Delete) operations, and the conversion of natural language to linux shell command statements. A vision for supporting enforcement models is explored, in addition to a self-training model that may be effective in correcting problematic behavior when encountered. 

The initial goal was the creation of an LLM microservice architecture to translate user prompts to executable code while checking for policy violations, and detecting and preventing repeated attacks. As our research continued, we found challenges that required a pivot from initial expectations. We shifted to a focus on R/W/X (Read/Write/Execute) security properties, and condensing models that can generate json access statements from shell script one-liners.  

## Visualizations

[Schelling Segregation Model](https://github.com/1337kiwi/schelling_segregation_model)

The Schelling segregation model is a classic agent-based model, demonstrating how even a mild preference for similar neighbors can lead to a much higher degree of segregation than we would intuitively expect. The model consists of agents on a square grid, where each grid cell can contain at most one agent. Agents come in two colors: red and blue, and 3 shades, dark, light, and pale. They are happy if a certain number of their eight possible neighbors are of the same color, and unhappy otherwise. Unhappy agents will pick a random empty cell to move to each step, until they are happy. The model keeps running until there are no unhappy agents.

[Misinformation Propagation Model](https://github.com/1337kiwi/AI-MisinfoProp)

This model is built similarly to the previous one, but is the implementation of an agent-based simulation model designed to study the spread of misinformation within a community. The model includes both human agents and language model (LLM) agents, which can be either benign or malicious. The simulation tracks how agents interact, move, and influence each other's beliefs over time.

To simulate interactions between individuals, we select 3 discrete random agents within another agent’s direct community (moore neighborhood), and compute the information value they receive,  to be
Θ = ∑ α(β + ϵ)
In which α is a belief of a neighbor within a set of 3 neighbors randomly sampled from the direct moore neighborhood, β is the confidence level of the discrete neighbor, and ϵ is the trust coefficient held by the agent deciding on the information value, which can be either the trust coefficient towards models or humans, depending on the neighbor’s agent type. Humans have consistent trust coefficients towards other humans, but initialize with variable trust coefficients towards models, depending on the type of agent they are.

This simulation also allows one to determine if there is a specific “turning point” in which the majority of information propagation becomes misinformed, and the trends that occur as a result. By varying the proportion of models to humans, as well as malicious models within that subset, we can model how humans in social media environments filled with models can be affected.

## Sites 

[PhotonVision Site](https://photonvision.org/)

A cute lil site for a cute lil FRC Program.

## Panels

SecureIT Summer Camp Panel (2023, 2024)