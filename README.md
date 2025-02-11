<h1 align="center">
  <a href="https://github.com/dec0dOS/amazing-github-template">
    <img src="https://i.imgur.com/TBRZwDu.png" alt="Logo" width="800" height="200">
  </a>
</h1>

<div align="center">
LAVIS's InstructBLIP model finetuned to remote sensing image-text data via Reinforcement Learning. The aim is to teach Visual Reasoning to a VLM on Remote Sensing imagery, which is only scarcely present in its pretraining dataset.
</div>

<div align="center">
<br />

[![license](https://img.shields.io/github/license/dec0dOS/amazing-github-template.svg?style=flat-square)](LICENSE)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
  - [Built With](#built-with)
  - [Figure](#figure-model)
  - [Qualitative results](#quali)
  - [Quantitative results](#quant)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

---

## About

<table>
<tr>
<td>

Forked from SalesForce's LAVIS repository, this improved version implements Reinforcement Learning to bolster image captioning abilities for the specific domain of remote sensing. On top of optimization through Cross-Entropy loss minimization, a few supplementary Reinforcement Learning epochs are completed to guide the model towards more desirable outputs, using learning signals tailored to the domain of Remote Sensing. More precisely, **Self-Critical Sequence Training** (<a>https://arxiv.org/abs/1612.00563), a variant of the **REINFORCE** algorithm which is similar to PPO, is used to enforce these learning signals.  

<details open>
<summary>Additional info</summary>
<br>
Note that **SCST** can be made compatible with PPO/GRPO, with the issue that there are no intermediate rewards during the generation of a caption (the full generated caption is required to compute the learning signals).
</details>

</td>
</tr>
</table>

### Built With

- [SalesForce's LAVIS repository](https://github.com/salesforce/LAVIS)
This repository relies almost entirely on LAVIS; a few modifications allows it to be finetuned using RL.
- [FACTUAL scene graph extractor](https://github.com/zhuang-li/FactualSceneGraph)
One of the most impactful reward function is obtained by measuring the closeness of generated captions and ground-truth (human-annotated) captions. FACTUAL extracts "scene graphs", like the SPICE metric, to compute such a reward by comparing the graphs. It also highlights the missing objects and the hallucinations made by the model.

**Figure**

<img src="https://i.imgur.com/AasnyVG.png" alt="Figure BLIP_SCST" width="1000" height="300">

## Getting Started

### Prerequisites

- Clone the present repository (installing LAVIS from scratch will require multiple precise modifications in the code that have already been done in this very repository).
- Reward functions utilities:
**FACTUAL Scene Graph Extraction**
```sh
pip install FactualSceneGraph
```
OR choose a pretrained model from huggingface: <a> https://github.com/zhuang-li/FactualSceneGraph </a>


### Usage

#### RSRL-LAVIS

After installing this repository, you need to create an environment, activate it, and install the libraries from requirements.txt. **PYTHON 3.9+ REQUIRED**

**conda**
```sh
conda create --name lavis_rl python=3.9
conda activate lavis_rl
```

**pip**
```sh
pip install -r requirements.txt
```

#### Best model

weights for the best InstructBLIP model I have managed to obtain.
<a>https://huggingface.co/tdujardin/InstructBLIP_RS_RL/tree/main</a>

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE) for more information.

## Acknowledgements

Thanks to SalesForce and their **BLIP** repository that made it simple to implement RL algorithms, and to train on my own data.
