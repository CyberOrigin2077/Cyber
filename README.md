# CYBER: A General Robotic Operation System for Embodied AI

![Show Data](docs/imgs/showdata.png)

The development of world models in robotics has long been a cornerstone of advanced research, with most approaches relying heavily on vast, platform-specific datasets. These datasets, while valuable, often limit scalability and generalization to different robotic platforms, restricting their broader applicability.

In contrast, **CYBER** approaches world modeling from a "first principles" perspective, drawing inspiration from how humans naturally acquire skills through experience and interaction with their environment. **CYBER** is the first general Robotic Operational System designed to adapt to both teleoperated manipulation and human operation data, enabling robots to learn and predict across a wide range of tasks and environments. It builds with a <u>Physical World Model</u>, a cross-embodied <u>Visual-Language Action Model</u> (VLA), a <u>Perception Model</u>, a <u>Memory Model</u>, and a <u>Control Model</u> to help robots learn, predict, and memory across various tasks and embodiments.
In contrast, **CYBER** approaches world modeling from a "first principles" perspective, drawing inspiration from how humans naturally acquire skills through experience and interaction with their environment. **CYBER** is the first general Robotic Operational System designed to adapt to both teleoperated manipulation and human operation data, enabling robots to learn and predict across a wide range of tasks and environments. It builds with a <u>Physical World Model</u>, a cross-embodied <u>Visual-Language Action Model</u> (VLA), a <u>Perception Model</u>, a <u>Memory Model</u>, and a <u>Control Model</u> to help robots learn, predict, and memory across various tasks and embodiments.

At the same time, **CYBER** also provide millions of human operation datasets and baseline models over HuggingFace 🤗 to enhance embodied learning, and experimental evalaution tool box to help researchers to test and evaluate their models in both simulation and real world.

---

## 🌟 Key Features

- **🛠️ Modular**: Built with a modular architecture, allowing flexibility in various environments.
- **📊 Data-Driven**: Leverages millions of human operation datasets to enhance embodied learning.
- **📈 Scalable**: Scales across different robotic platforms, adapting to new environments and tasks.
- **🔧 Customizable**: Allows for customization and fine-tuning to meet specific requirements.
- **📚 Extensible**: Supports the addition of new modules and functionalities, enhancing capabilities.
- **📦 Open Source**: Open-source and freely available, fostering collaboration and innovation.
- **🔬 Experimental**: Supports experimentation and testing, enabling continuous improvement.
---

## 🛠️ Modular Components

**CYBER** is built with a modular architecture, allowing for flexibility and customization. Here are the key components:

- [**🌍 World Model**](docs/tutorial/world.md): Learns from physical interactions to understand and predict the environment.
- [**🎬 Action Model**](docs/tutorial/action.md): Learns from actions and interactions to perform tasks and navigate.
- [**👁️ Perception Model**](docs/tutorial/preception.md): Processes sensory inputs to perceive and interpret surroundings.
- [**🧠 Memory Model**](docs/tutorial/memory.md): Utilizes past experiences to inform current decisions.
- [**🎮 Control Model**](docs/tutorial/control.md): Manages control inputs for movement and interaction.

**🌍 World Model** is now available. Additional models will be released soon.

**🌍 World Model** is now available. Additional models will be released soon.

## ⚙️ Setup

### Pre-requisites

You will need Anaconda installed on your machine. If you don't have it installed, you can follow the installation instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

### Installation

You can run the following commands to install CYBER:

```bash
bash scripts/build.sh
```

Alternatively, you can install it manually by following the steps below:

1. **Create a clean conda environment:**

        conda create -n cyber python=3.10 && conda activate cyber

2. **Install PyTorch and torchvision:**

        conda install pytorch==2.3.0 torchvision==0.18.0 cudatoolkit=11.1 -c pytorch -c nvidia

3. **Install the CYBER package:**

        pip install -e .

## 🤗 Hugging Face Integration

**CYBER** leverages the power of Hugging Face for model sharing and collaboration. You can easily access and use our models through the Hugging Face platform.

### Available Data

Currently, four tasks are available for download:
Currently, four tasks are available for download:

- 🤗 [Pipette](https://huggingface.co/datasets/cyberorigin/cyber_pipette): Bimanual human demonstration dataset of precision pipetting tasks for laboratory manipulation.
- 🤗 [Take Item](https://huggingface.co/datasets/cyberorigin/cyber_take_the_item): Single-arm manipulation demonstrations of object pick-and-place tasks.
- 🤗 [Twist Tube](https://huggingface.co/datasets/cyberorigin/cyber_twist_the_tube): Bimanual demonstration dataset of coordinated tube manipulation sequences.
- 🤗 [Fold Towels](https://huggingface.co/datasets/cyberorigin/cyber_fold_towels): Bimanual manipulation demonstrations of deformable object folding procedures.
- 🤗 [Pipette](https://huggingface.co/datasets/cyberorigin/cyber_pipette): Bimanual human demonstration dataset of precision pipetting tasks for laboratory manipulation.
- 🤗 [Take Item](https://huggingface.co/datasets/cyberorigin/cyber_take_the_item): Single-arm manipulation demonstrations of object pick-and-place tasks.
- 🤗 [Twist Tube](https://huggingface.co/datasets/cyberorigin/cyber_twist_the_tube): Bimanual demonstration dataset of coordinated tube manipulation sequences.
- 🤗 [Fold Towels](https://huggingface.co/datasets/cyberorigin/cyber_fold_towels): Bimanual manipulation demonstrations of deformable object folding procedures.

### Available Models

Our pretrained models will be released on Hugging Face soon:

- Cyber-World-Large (Coming Soon)
- [Cyber-World-Base](https://huggingface.co/cyberorigin/GENIE_Base)

- Cyber-World-Small (Coming Soon)

### Using the Models (Coming Soon)

<!-- To use our models in your project, you can install the `transformers` library and load the models as follows:
### Using the Models (Coming Soon)

<!-- To use our models in your project, you can install the `transformers` library and load the models as follows:

```python
from transformers import AutoModel, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cyberorigin/cyber-base")
model = AutoModel.from_pretrained("cyberorigin/cyber-base")

# Example usage
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

For more details, please refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/index). -->
For more details, please refer to the [Hugging Face documentation](https://huggingface.co/docs/transformers/index). -->


## 🕹️ Usage

Please refer to the [experiments](docs/experiments/world_model.md) for more details on data downloading and model training.

---

## 💾 File Structure

```plaintext
├── ...
├── docs                   # documentation files and figures 
├── docker                 # docker files for containerization
├── examples               # example code snippets
├── tests                  # test cases and scripts
├── scripts                # scripts for setup and utilities
├── experiments            # model implementation and details
│   ├── configs            # model configurations
│   ├── models             # model training and evaluation scripts
│   ├── notebooks          # sample notebooks
│   └── ...
├── cyber                  # compression, model training, and dataset source code
│   ├── dataset            # dataset processing and loading
│   ├── utils              # utility functions
│   └── models             # model definitions and architectures
│       ├── action         # visual language action model
│       ├── control        # robot platform control model
│       ├── memory         # lifelong memory model
│       ├── perception     # perception and scene understanding model
│       ├── world          # physical world model
│       └── ...
└── ...
```

### 📕 References

[Magvit2](https://github.com/TencentARC/Open-MAGVIT2) and [GENIE](https://arxiv.org/abs/2402.15391) adapted from [1xGPT Challenge](https://github.com/1x-technologies/1xgpt)
[Magvit2](https://github.com/TencentARC/Open-MAGVIT2) and [GENIE](https://arxiv.org/abs/2402.15391) adapted from [1xGPT Challenge](https://github.com/1x-technologies/1xgpt)
1X Technologies. (2024). 1X World Model Challenge (Version 1.1) [Data set]


```bibtex
@inproceedings{wang2024hpt,
author    = {Lirui Wang, Xinlei Chen, Jialiang Zhao, Kaiming He},
title     = {Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers},
booktitle = {Neurips},
year      = {2024}
}
```
```bibtex
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}
```

## 📄 Dataset Metadata
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">CyberOrigin Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/CyberOrigin2077/Cyber</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">Cyber represents a model implementation that seamlessly integrates state-of-the-art (SOTA) world models with the proposed CyberOrigin Dataset, pushing the boundaries of artificial intelligence and machine learning.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CyberOrigin</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Apache 2.0</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>

## 📫 Contact
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}
```

## 📄 Dataset Metadata
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">CyberOrigin Dataset</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/CyberOrigin2077/Cyber</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">Cyber represents a model implementation that seamlessly integrates state-of-the-art (SOTA) world models with the proposed CyberOrigin Dataset, pushing the boundaries of artificial intelligence and machine learning.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CyberOrigin</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Apache 2.0</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>

## 📫 Contact
If you have technical questions, please open a GitHub issue. For business development or other collaboration inquiries, feel free to contact us through email 📧 (<contact@cyberorigin.ai>). Enjoy! 🎉
