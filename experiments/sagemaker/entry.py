import os

if __name__ == "__main__":
    # Install cyber
    os.system("pip install -e .")

    # Download data from huggingface
    os.system("huggingface-cli download cyberorigin/cyber_pipette --repo-type dataset --local-dir data/cyber_pipette")

    # Invoke Training Script
    os.system("chmod +x ./train_script_sagemaker.sh")
    os.system("/bin/bash -c ./train_script_sagemaker.sh")
