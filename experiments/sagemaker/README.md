# Training Using Sagemaker

This folder contains the additional config files and scripts to train the GENIE model on AWS Sagemaker.

- accelerate_config.yaml:  Accelerate configuration file for distributed training.
- entry.py: Entry point for the training job.
- README.md: Instructions for setting up the training job.
- run.ipynb: Jupyter notebook for queueing the training job.
- train_script_sagemaker.sh: Script to invoke the training script using accelerate.

## Prerequisites
- AWS account with Sagemaker. You can follow the [official guide](https://aws.amazon.com/tutorials/machine-learning-tutorial-train-a-model/) if you want to know more about SM training jobs and verify your account settings.
- WandB account for logging. You can create an account [here](https://wandb.ai/authorize).
- Credits. (1 hour of training on p4.xlarge instance costs around $30 :D)

## Training

**1.** Open a notebook instance on sagemaker. You can follow the [official guide](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html) to create a notebook instance.

**2.** Create a terminal in the notebook instance. Run the following commands to create a working directory and clone this repo.
```bash
mkdir training
cd training
git clone https://github.com/CyberOrigin2077/Cyber.git
```
After this step your folder structure should look like this:
```
training
└── Cyber
    ├── experiments
    |   ├── sagemaker
    |   │   ├── accelerate_config.yaml
    |   │   ├── entry.py
    |   │   ├── README.md
    |   │   ├── run.ipynb
    |   │   └── train_script_sagemaker.sh
    ...
```
**3.** Set up the folder structure by running the following commands:
```bash
cp -r Cyber/experiments/sagemaker/* Cyber/
mv Cyber/run.ipynb .
```
After this step your folder structure should look like this:
```
training
├── Cyber
|   ├── accelerate_config.yaml
|   ├── entry.py
|   ├── README.md
|   ├── train_script_sagemaker.sh
|   └── ...
└── run.ipynb
    ...
```

**4.** Install and authenticate WandB. Run the following commands in the terminal.
```bash
pip install wandb
wandb login
```

**5.** Open the `run.ipynb` notebook and follow the instructions to queue the training job. If you are successful, your job should start running on Sagemaker. If you used all default settings, finishing 1 epoch should take around 4 hours, which would set you back around $120. Enjoy! :D