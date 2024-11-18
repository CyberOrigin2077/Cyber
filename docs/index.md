# CYBER: A General Robotic Operation System for Embodied AI

![Show Data](imgs/showdata.png)

The development of world models in robotics has long been a cornerstone of advanced research, with most approaches relying heavily on vast, platform-specific datasets. These datasets, while valuable, often limit scalability and generalization to different robotic platforms, restricting their broader applicability.

In contrast, **CYBER** approaches world modeling from a "first principles" perspective, drawing inspiration from how humans naturally acquire skills through experience and interaction with their environment. **CYBER** is the first general Robotic Operational System designed to adapt to both teleoperated manipulation and human operation data, enabling robots to learn and predict across a wide range of tasks and environments. It builds with a <u>Physical World Model</u>, a cross-embodied <u>Visual-Language Action Model</u> (VLA), a <u>Perception Model</u>, a <u>Memory Model</u>, and a <u>Control Model</u> to help robots learn, predict, and memory across various tasks and embodiments.

At the same time, **CYBER** also provide millions of human operation datasets and baseline models over HuggingFace 🤗 to enhance embodied learning, and experimental evalaution tool box to help researchers to test and evaluate their models in both simulation and real world.

---

## 🛠️ Modular Components

**CYBER** is built with a modular architecture, allowing for flexibility and customization. Here are the key components:

- [**🌍 World Model**](tutorial/world.md): Learns from physical interactions to understand and predict the environment.
- [**🎬 Action Model**](tutorial/action.md): Learns from actions and interactions to perform tasks and navigate.
- [**👁️ Perception Model**](): Processes sensory inputs to perceive and interpret surroundings.
- [**🧠 Memory Model**](): Utilizes past experiences to inform current decisions.
- [**🎮 Control Model**](): Manages control inputs for movement and interaction.
---

## 📰 Release
- **2024-11-18:** **🌍 World Model** supports new tokenizer model [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) and new dynamic model [Deep Planning Network](https://github.com/-google-research/planet)
- **2024-10-23:** **🌍 World Model** is now available. Additional models will be released soon.
---
