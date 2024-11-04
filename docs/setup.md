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