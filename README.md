<h1 align="center">
  <a href="https://github.com/dec0dOS/amazing-github-template">
    <img src="https://github.com/user-attachments/assets/e8e84f28-a734-47a1-8239-e90a0b74a408" alt="Logo" width="800" height="200">
  </a>
</h1>

<div align="center">
  LAVIS's InstructBLIP model finetuned to remote sensing image-text data via Reinforcement Learning.
</div>

<div align="center">
<br />

[![license](https://img.shields.io/github/license/dec0dOS/amazing-github-template.svg?style=flat-square)](LICENSE)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Cookiecutter template](#cookiecutter-template)
    - [Manual setup](#manual-setup)
    - [Variables reference](#variables-reference)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Support](#support)
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
- [FACTUAL scene graph extractor](https://github.com/zhuang-li/FactualSceneGraph)

## Getting Started

### Prerequisites

- Clone the present repository (installing LAVIS from scratch will require multiple precise modifications in the code that have already been done in this very repository).
- Reward functions utilities:
**FACTUAL Scene Graph Extraction**
```sh
pip install FactualSceneGraph
```
OR
[Thehttps://github.com/zhuang-li/FactualSceneGraph


### Usage

#### Cookiecutter template

After installing Cookiecutter, all you need to do is to run the following command:

```sh
cookiecutter gh:dec0dOS/amazing-github-template
```

You will get an interactive prompt where you'll specify relevant options for your project (or the default value will be used).

![Preview](docs/images/preview.svg)

#### Manual setup

Please follow these steps for manual setup:

1. [Download the precompiled template](https://github.com/dec0dOS/amazing-github-template/releases/download/latest/template.zip)
2. Replace all the [variables](#variables-reference) to your desired values
3. Initialize the repo in the precompiled template folder

    `or`

    Move the necessary files from precompiled template folder to your existing project directory. Don't forget the `.github` directory that may be hidden by default in your operating system

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE) for more information.

## Acknowledgements

Thanks for these awesome resources that were used during the development of the **Amazing GitHub template**:
