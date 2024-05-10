# Exploring the Adversarial Capabilities of Large Language Models

  <center>
  <img src="images/concept.png" alt="Concept"  height=250>
  </center>

> **Abstract:**
> *The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.*  
[Full Paper](https://arxiv.org/abs/2402.09132)


# Setup and Run Attacks

## Setup Docker Container
The easiest way to perform the attacks is to run the code in a Docker container. To build the Docker image, run the following script:

```bash
docker build -t adv_llm  .
```

To create and start a Docker container, run the following command from the project's root:

```bash
docker run --rm --shm-size 16G --name my_container --gpus '"device=all"' -v $(pwd):/workspace -it adv_llm bash
```

## Run the Attack
To craft adversarial examples with an LLM, simply run the following script:
```bash
python craft_adv_examples.py
```
Hyperparameters, input data, and the models can be specified directly in the script. All attack results will automatically be written into a TSV file for evaluation.


## Citation
If you build upon our work, please don't forget to cite us.
```
@article{struppek24adversarialllm,
  author = {Struppek, Lukas and Le, Minh Hieu and Hintersdorf, Dominik and Kersting, Kristian},
  title = {Exploring the Adversarial Capabilities of Large Language Models},
  journal = {International Conference on Learning Representations (ICLR) - Workshop on Secure and Trustworthy Large Language Models},
  year = {2024},
}
```