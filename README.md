Project 2B – Full Workflow Documentation



You can find the required database of csv files at this link
https://www.kaggle.com/datasets/wajahat1064/emotion-recognition-using-eeg-and-computer-games/data

1. Setting Up a FABRIC Node

Logged into the FABRIC portal.

Created a new slice and added one compute node.

Selected:

Ubuntu 22.04 image

Sufficient CPU cores

At least 10 GB storage

Submitted the slice request.

Waited until the slice status showed Active.

Downloaded the generated SSH configuration file (fabric_ssh_config).

2. Connecting to the FABRIC Node

From my local machine (Windows PowerShell), I connected using:

ssh -F "C:\Users\Cdelt\Downloads\fabric_ssh_config" eegnode

This uses the custom SSH configuration file provided by FABRIC.

After connecting, the prompt showed:

ubuntu@eegnode:~$
3. Creating and Activating Conda Environment

Inside the node:

conda create -n eeg_env python=3.10 -y
conda activate eeg_env

Installed required packages:

conda install -y -c conda-forge numpy pandas scikit-learn matplotlib
conda install -y -c pytorch pytorch cpuonly

Verified installation:

python -c "import torch, pandas, numpy"
4. Uploading the Dataset

Uploaded the GAMEEMO dataset to:

~/data/GAMEEMO

Files were transferred using scp from the local machine:

scp -F "C:\Users\Cdelt\Downloads\fabric_ssh_config" -r GAMEEMO eegnode:~/data/

Confirmed upload:

ls ~/data/GAMEEMO
5. Writing the Training Script

Created the script:

nano ~/work/project2b_increment_analysis.py

The script performs:

Reading and preprocessing all subjects

Selecting 14 EEG channels

Windowing data (256 samples, stride 256)

Subject-wise train/test split using GroupShuffleSplit

Standardizing features (train only)

Training multiple MLP architectures

Recording:

Training accuracy

Test accuracy

Training runtime

Evaluation runtime

Total runtime

Saving results to CSV

Generating 4 required plots

Saved and exited nano:

Ctrl + O → Enter → Ctrl + X
6. Running the Script

Activated environment:

conda activate eeg_env

Ran:

python ~/work/project2b_increment_analysis.py --data_root ~/data/GAMEEMO --epochs 6 --win 256 --stride 256

The script trained 6 models:

[32]

[64]

[128]

[256]

[256, 128]

[512, 256, 128]

Each increment increased network capacity.

7. Output Generated

The script produced:

project2b_increment_table.csv

plot_train_accuracy.png

plot_test_accuracy.png

plot_train_runtime.png

plot_total_runtime.png

8. Downloading Results to Local Machine

Exited SSH:

exit

Downloaded files using:

\Desktop\
