#!/usr/bin/env python
# coding: utf-8

# # Setup
# -  Follow the setup instructions based on your preferred environment!

# ## Local

# One of our key goals in designing this assignment is to allow you to complete most of the preliminary implementation work locally.  
# We highly recommend that you **pass all tests locally** using the provided `hw4_data_subset` before moving to a GPU runtime.  
# To do this, simply:
# 
# ### Create a new conda environment
# ```bash
# # Be sure to deactivate any active environments first
# conda create -n hw4 python=3.12.4
# ```
# 
# ### Activate the conda environment
# ```bash
# conda activate hw4
# ```
# 
# ### Install the dependencies using the provided `requirements.txt`
# ```bash
# pip install --no-cache-dir --ignore-installed -r requirements.txt
# ```
# 
# ### Ensure that your notebook is in the same working directory as the `Handout`
# This can be achieved by:
# 1. Physically moving the notebook into the handout directory.
# 2. Changing the notebook’s current working directory to the handout directory using the os.chdir() function.
# 
# ### Open the notebook and select the newly created environment from the kernel selector.
# 
# If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:
# ```
# .
# ├── README.md
# ├── requirements.txt
# ├── hw4lib/
# ├── mytorch/
# ├── tests/
# └── hw4_data_subset/
# ```

# ## Colab

# ### Step 1: Get your handout
# - See writeup for recommended approaches.

# # Example: My preferred approach
# import os
# # Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
# os.environ['GITHUB_TOKEN'] = "your_token_here"
# 
# GITHUB_USERNAME = "your_username_here"
# REPO_NAME       = "your_repo_name_here"
# TOKEN = os.environ.get("GITHUB_TOKEN")
# repo_url        = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
# !git clone {repo_url}

# # To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
# !cd {REPO_NAME} && git pull

# ### Step 2: Install Dependencies
# - `NOTE`: Your runtime will be restarted to ensure all dependencies are updated.
# - `NOTE`: You will see a runtime crashed message, this was intentionally done. Simply move on to the next cell.

# %pip install --no-deps -r IDL-HW4/requirements.txt
# import os
# os.kill(os.getpid(), 9) # NOTE: This will restart the your colab Python runtime (required)!

# ### Step 3: Obtain Data
# 
# - `NOTE`: This process will automatically download and unzip data for both `HW4P1` and `HW4P2`.  
# 

# !curl -L -o /content/s25-hw4-data.zip https://www.kaggle.com/api/v1/datasets/download/cmu11785/s25-hw4-data
# !unzip -q -o /content/s25-hw4-data.zip -d /content/hw4_data
# !rm -rf /content/s25-hw4-data.zip
# !du -h --max-depth=2 /content/hw4_data

# ### Step 4: Move to Handout Directory
# You must be within the handout directory for the library imports to work!
# 
# - `NOTE`: You may have to repeat running this command anytime you restart your runtime.
# - `NOTE`: You can do a `pwd` to check if you are in the right directory.
# - `NOTE`: The way it is setup currently, Your data directory should be one level up from your project directory. Keep this in mind when you are setting your `root` in the config file.
# 
# If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:
# ```
# .
# ├── README.md
# ├── requirements.txt
# ├── hw4lib/
# ├── mytorch/
# ├── tests/
# └── hw4_data_subset/
# 
# ```

# import os
# os.chdir('IDL-HW4')
# !ls

# ## Kaggle

# While it is possible to run the notebook on Kaggle, we would recommend against it. This assignment is more resource intensive and may run slower on Kaggle. 

# ### Step 1: Get your handout
# - See writeup for recommended approaches.

# # Example: My preferred approach
# import os
# # Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
# os.environ['GITHUB_TOKEN'] = "your_token_here"
# 
# GITHUB_USERNAME = "your_username_here"
# REPO_NAME       = "your_repo_name_here"
# TOKEN = os.environ.get("GITHUB_TOKEN")
# repo_url        = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
# !git clone {repo_url}

# # To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
# !cd {REPO_NAME} && git pull

# ### Step 2: Install Dependencies
# - Simply set the `Environment` setting in the notebook to `Always use latest environment`. No need to install anything.

# ### Step 3: Obtain Data
# 
# #### ⚠️ Important: Kaggle Users  
# If you are using Kaggle, **do not manually download the data!** The dataset is large and may exceed your available disk space. Instead, follow these steps to add the dataset directly to your notebook:
# 
# 1. Open your **Kaggle Notebook**.  
# 2. Navigate to **Notebook → Input**.  
# 3. Click **Add Input**.  
# 4. In the search bar, paste the following URL:  
#    👉 [https://www.kaggle.com/datasets/cmu11785/s25-hw4-data](https://www.kaggle.com/datasets/cmu11785/s25-hw4-data)  
# 5. Click the **➕ (plus sign)** to add the dataset to your notebook.  
# 
# #### 📌 Note:  
# This process will automatically download and unzip data for both `HW4P1` and `HW4P2`.  
# 

# ### Step 4: Move to Handout Directory
# You must be within the handout directory for the library imports to work!
# 
# - `NOTE`: You may have to repeat running this command anytime you restart your runtime.
# - `NOTE`: You can do a `pwd` to check if you are in the right directory.
# - `NOTE`: The way it is setup currently, Your data directory should be one level up from your project directory. Keep this in mind when you are setting your `root` in the config file.
# 
# If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:
# ```
# .
# ├── README.md
# ├── requirements.txt
# ├── hw4lib/
# ├── mytorch/
# ├── tests/
# └── hw4_data_subset/
# 
# ```

# import os
# os.chdir('IDL-HW4')
# !ls

# ## PSC

# ### Step 1: Get your handout
# - See writeup for recommended approaches.
# - If you use Remote - SSH to connect to Bridges2, you can upload the handout to your project directory and work from there.
# 

# # Example: My preferred approach
# import os
# # Settings -> Developer Settings -> Personal Access Tokens -> Token (classic)
# os.environ['GITHUB_TOKEN'] = "your_token_here"
# 
# GITHUB_USERNAME = "your_username_here"
# REPO_NAME       = "your_repo_name_here"
# TOKEN = os.environ.get("GITHUB_TOKEN")
# repo_url        = f"https://{TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
# !git clone {repo_url}

# # To pull latest changes (Must be in the repo dir, use pwd/ls to verify)
# !cd {REPO_NAME} && git pull

# ### Step 2: Setting Up Your Environment on Bridges2
# 
# For this homework, we are providing a shared Conda environment for the entire class. Follow these steps to set up the environment and start a Jupyter notebook on Bridges2:

# #### 1. SSH into Bridges2
# ```bash
# ssh username@bridges2.psc.edu
# ```

# #### 2. Navigate to your Project Directory
# ```bash
# cd $PROJECT
# ```

# #### 3. Load the Anaconda Module
# ```bash
# module load anaconda3
# ```

# #### 4. Activate the provided HW4 Environment
# ```bash
# conda deactivate # First, deactivate any existing Conda environment
# conda activate /jet/home/psamal/hw_envs/idl_hw4
# ```

# #### 5. Request a Compute Node
# ```bash
# interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00
# ```

# #### 6. Re-activate Environment
# If your Conda environment was deactivated due to node allocation:
# ```bash
# conda deactivate # First, deactivate any existing Conda environment
# conda activate /jet/home/psamal/hw_envs/idl_hw4
# ```

# #### 7. Start Jupyter Notebook
# Launch Jupyter Notebook:
# ```bash
# jupyter notebook --no-browser --ip=0.0.0.0
# ```

# 
# #### 8. Connect to Jupyter Server
# 
# You can now use your prefered way of connecting to the Jupyter Server. Your options should be covered in the docs linked in post 558 @ piazza.
# 
# The following is my preferred way of connecting to the Jupyter Server:
# 
# ##### 8.1 Connect in VSCode
# I prefer uploading the notebook to PSC Bridges2 storage ($PROJECT directory) and then connecting to the Jupyter Server from there.
# 1. Use Remote - SSH to connect to Bridges2 and navigate to your project directory.
# 2. Upload the notebook to the project directory.
# 3. Open the notebook in VSCode.
# 4. Go to **Kernel** → **Select Another Kernel** → **Existing Jupyter Server**
# 5. Enter the URL of the Jupyter Server:```http://{hostname}:{port}/tree?token={token}```
#    - eg: `http://v011.ib.bridges2.psc.edu:8888/tree?token=e4b302434e68990f28bc2b4ae8d216eb87eecb7090526249` 
# 
# > **Note**: Replace `{hostname}`, `{port}` and `{token}` with your actual values from the Jupyter output.

# ### Step 3: Get Data
# - `NOTE`: This will download and unzip data for both `HW4P1` and `HW4P2`
# - `NOTE`: We are using `$LOCAL`: the scratch storage on local disk on the node running a job to store out data. 
#   - Disk accesses are much faster than what you would get from `$PROJECT` storage
#   - `IT IS NOT PERSISTENT`
# - `NOTE`: Make sure you have completed the previous steps before running this cell.
# - Read more about it PSC File Spaces [here](https://www.psc.edu/resources/bridges-2/user-guide#file-spaces).

# !curl -L -o $LOCAL/s25-hw4-data.zip https://www.kaggle.com/api/v1/datasets/download/cmu11785/s25-hw4-data
# !unzip -q -o $LOCAL/s25-hw4-data.zip -d $LOCAL/hw4_data
# !rm -rf $LOCAL/s25-hw4-data.zip\
# !du -h --max-depth=2 $LOCAL/hw4_data

# ### Step 4: Move to Handout Directory
# Depending on the way you are running your notebook, you may or may not need to run this cell. As long as you are within the handout directory for the library imports to work!
# 
# - `NOTE`: You may have to repeat running this command anytime you restart your runtime.
# - `NOTE`: You can do a `pwd` to check if you are in the right directory.
# - `NOTE`: The way it is setup currently, Your data directory should be one level up from your project directory. Keep this in mind when you are setting your `root` in the config file.
# 
# If everything was done correctly, You should see atleast the following files in your current working directory after running `!ls`:
# ```
# .
# ├── README.md
# ├── requirements.txt
# ├── hw4lib/
# ├── mytorch/
# ├── tests/
# └── hw4_data_subset/
# 
# ```

# # Move to the handout directory if you are not there already
# import os
# os.chdir('IDL-HW4')
# !ls

# # Imports
# 
# - If your setup was done correctly, you should be able to run the following cell without any issues.

# In[1]:


from hw4lib.data import (
    H4Tokenizer,
    LMDataset,
    verify_dataloader
)
from hw4lib.model import (
    CausalMask,
    PadMask,
    PositionalEncoding,
    DecoderOnlyTransformer
)
from hw4lib.utils import (
    create_optimizer,
    create_scheduler,
    plot_lr_schedule
)
from hw4lib.trainers import (
    LMTrainer,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import gc
import torch
from torchinfo import summary
import os
import json
import tarfile
import shutil
import wandb
import yaml
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# # Implementations
# 
# - `NOTE`: All of these implementations have detailed specification, implementation details, and hints in their respective source files. Make sure to read all of them in their entirety to understand the implementation details!

# ## MyTorch Implementations
# - Modify your `Linear` implementation from HW1P1 to support arbitrary number of dimensions in `mytorch/nn/linear.py`.
# - Modify your `Softmax` implementation from HW1P1 to support arbitrary number of dimensions in `mytorch/nn/activation.py`.
# - Implement the `ScaledDotProductAttention` class in `mytorch/nn/scaled_dot_product_attention.py`.
# - Implement the `MultiHeadAttention` class in `mytorch/nn/multi_head_attention.py`.
# - Run the cell below to check your implementations.
# 

# !python -m tests.test_mytorch

# ## Dataset Implementation
# - Familiarize yourself with the `tokenize`, `encode`, and `decode` methods of the `H4Tokenizer` class in `hw4lib/data/tokenizer.py`. You will need to make use of these methods in both `HW4P1` and `HW4P2` both in the dataset implementations and during decoding.
# - Implement the `LMDataset` class in `hw4lib/data/lm_dataset.py`.
#     - You will have to implement parts of `__init__` and completely implement the `__len__`, `__getitem__` and `collate_fn` methods.
# - Run the cell below to check your implementation.
# 

# !python -m tests.test_dataset_lm

# ## Model Implementations
# #### Overview:
# - Implement the `CausalMask` and `PadMask` functions in `hw4lib/modules/masks.py` to handle masking.
# - Implement the `PositionalEncoding` class in `hw4lib/model/positional_encoding.py` to handle positional encoding.
# - Implement the Transformer Sublayers: `SelfAttentionLayer` and `FeedForwardLayer` classes in `hw4lib/model/sublayers.py`.
# - Implement the Transformer Layer: `SelfAttentionDecoderLayer` class in `hw4lib/model/decoder_layers.py`.
# - Implement the `DecoderOnlyTransformer` class in `hw4lib/model/transformers.py`.
# - Run the cells below to check your implementation.
# - `NOTE`: Besides the `DecoderOnlyTransformer` (P1 mandatory, P2 optional), you will use all of the above implementations in both `HW4P1` and `HW4P2`!

# ### Masks
# - Implement the `PadMask` and `CausalMask` functions in `hw4lib/modules/masks.py`.
# - Run the cell below to check your implementation.
# - You will need to make use of these masks in both `HW4P1` and `HW4P2`.

# #### Causal Mask

# !python -m tests.test_mask_causal

# #### Padding Mask

# !python -m tests.test_mask_padding

# #### Optional: Visualize your Masks

# # Dummy data
# _d_model   = 64
# _x         = torch.zeros(4, 20, _d_model)
# _x_len     = torch.tensor([5, 15, 10, 20])
# _x_causal  = CausalMask(_x)
# _x_padding = PadMask(_x, _x_len)
# 
# # Create figure with two subplots side by side
# fig, mask_axs = plt.subplots(1, 2, figsize=(12, 4))
# 
# # Plot masks
# masks_and_titles = [
#     (_x_padding, "Padding Mask"),
#     (_x_causal, "Causal Mask")
# ]
# 
# # Plot each mask
# images = []
# for i, (mask, title) in enumerate(masks_and_titles):
#     im = mask_axs[i].imshow(mask, cmap="gray", aspect='auto')
#     mask_axs[i].set_title(title, fontsize=8)
#     images.append(im)
# 
# # Add colorbar at the bottom
# fig.subplots_adjust(bottom=0.2)  # Make space for colorbar
# cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])  # [left, bottom, width, height]
# cbar = plt.colorbar(images[0], cax=cbar_ax, orientation='horizontal')
# cbar.ax.set_xlabel('Mask Values', labelpad=5, fontsize=8)
# cbar.set_ticks([0, 1])
# cbar.set_ticklabels(['Attend (0)', 'Ignore/Mask (1)'])
# cbar.ax.tick_params(labelsize=6)
# 
# plt.show()

# ### Positional Encoding
# - Implement the `PositionalEncoding` class in `hw4lib/model/positional_encoding.py`.
# - Run the cell below to check your implementation.
# - You will need to make use of this positional encoding in both `HW4P1` and `HW4P2`.

# !python -m tests.test_positional_encoding

# #### Optional: Visualize your Positional Encoding

# # Create sample positional encoding
# d_model = 64
# max_len = 100
# pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
# pe = pos_encoding.pe.squeeze(0).numpy()  # Remove batch dimension and convert to numpy
# 
# # Create figure with two subplots side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# 
# # Plot 1: Positional encoding matrix
# im = ax1.imshow(pe, aspect='auto', cmap='RdBu',
#                 extent=[0, d_model, max_len, 0])  # Flip y-axis to show position top-to-bottom
# plt.colorbar(im, ax=ax1, label='Encoding Value')
# ax1.set_xlabel('Dimension')
# ax1.set_ylabel('Position')
# ax1.set_title('Positional Encoding Matrix')
# ax1.grid(False)
# 
# # Plot 2: Sinusoidal patterns
# dimensions = [0, 15, 31, 47, 63]  # Plot first few dimensions
# for dim in dimensions:
#     ax2.plot(pe[:, dim], label=f'dim {dim}')
# ax2.set_xlabel('Position')
# ax2.set_ylabel('Encoding Value')
# ax2.set_title('Sinusoidal Patterns for Different Dimensions')
# ax2.legend()
# ax2.grid(True)
# 
# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

# ### Transformer Sublayers
# - Implement the Transformer Sublayers: `SelfAttentionLayer`, and `FeedForwardLayer` classes in `hw4lib/model/sublayers.py`.
# - Run the cell below to check your implementation.
# - You will need to make use of all of these sublayers in both `HW4P1` and `HW4P2`.

# !python -m tests.test_sublayer_selfattention

# !python -m tests.test_sublayer_feedforward

# ### Transformer Self-Attention Decoder Layer
# - Implement the Transformer Layer: `SelfAttentionDecoderLayer` class in `hw4lib/model/decoder_layers.py`.
# - Run the cell below to check your implementation.
# - You will need to make use of this sublayer in `HW4P2`.

# !python -m tests.test_decoderlayer_selfattention

# ### Decoder-Only Transformer
# 
# - Implement the `DecoderOnlyTransformer` class in `hw4lib/model/transformers.py`.
# - Run the cell below to check your implementation.
# - You will need to make use of in `HW4P1` and optionally `HW4P2`.

# !python -m tests.test_transformer_decoder_only

# ## Decoding Implementation
# - Implement the `generate_greedy` method of the `SequenceGenerator` class in `hw4lib/decoding/sequence_generator.py`.
# - Run the cell below to check your implementation.

# !python -m tests.test_decoding --mode greedy

# ## Trainer Implementation
# You will have to do some minor in-filling for the `LMTrainer` class in `hw4lib/trainers/lm_trainer.py` before you can use it.
# - Fill in the `TODO`s in the `__init__`.
# - Fill in the `TODO`s in the `_train_epoch`.
# - Fill in the `TODO`s in the `_validate_epoch`.
# - Fill in the `TODO`s in the `generate` method.
# - Fill in the `TODO`s in the `train` method.
# 
# `WARNING`: There are no test's for this. Implement carefully!

# # Experiments
# From this point onwards you may want to switch to a `GPU` runtime. 
# - `OBJECTIVE`: You must achieve a per-character perplexity ≤ 3.5 in order to get points for Task 2.

# ## Config
# - You can use the `config.yaml` file to set your config for your ablation study.
# 
# ---
# ### Notes:
# 
# - Set `tokenization: token_type:` to specify your desired tokenization strategy
# - You will need to set the root path to your `hw4p1_data` folder in `data: root:`. This will depend on your setup. For eg. if you are following out setup instruction:
#   - `PSC`: `"/local/hw4_data/hw4p1_data"`
#   - `Colab:`: `"/content/hw4_data/hw4p1_data"`
#   - `Kaggle:`: `"/kaggle/input/s25-hw4-data/hw4p1_data"`
# - There's extra configurations in the `optimizer` section which will only be relevant if you decide to use the `create_optimizer` function we've provided in `hw4lib/utils/create_optimizer.py`.
# - `BE CAREFUL` while setting numeric values. Eg. `1e-4` will get serialized to a `str` while `1.0e-4` gets serialized to float. 
# 

# In[ ]:


#get_ipython().run_cell_magic('writefile', 'config_lm.yaml', '\nName                      : "lenghanz"\n\n###### Tokenization ------------------------------------------------------------\ntokenization:\n  token_type                : "10k"       # [char, 1k, 5k, 10k]\n  token_map :\n      \'char\': \'hw4lib/data/tokenizer_jsons/tokenizer_char.json\'\n      \'1k\'  : \'hw4lib/data/tokenizer_jsons/tokenizer_1000.json\'\n      \'5k\'  : \'hw4lib/data/tokenizer_jsons/tokenizer_5000.json\'\n      \'10k\' : \'hw4lib/data/tokenizer_jsons/tokenizer_10000.json\'\n\n###### Dataset -----------------------------------------------------------------\ndata:                    # Currently setup for Colab assuming out setup\n  root                 : "hw4_data/hw4p1_data"  # TODO: Set the root path of your data\n  train_partition      : "train"  # train\n  val_partition        : "val"    # val\n  test_partition       : "test"   # test\n  subset               : 1.0      # Load a subset of the data (for debugging, testing, etc\n  batch_size           : 1024      #\n  NUM_WORKERS          : 32        # Set to 0 for CPU\n\n###### Network Specs -------------------------------------------------------------\nmodel: # Decoder-Only Language Model (HW4P1)\n  d_model                   : 256\n  d_ff                      : 1024\n  num_layers                : 4\n  num_heads                 : 8\n  dropout                   : 0.2\n  layer_drop_rate           : 0.1\n  weight_tying              : False\n\n###### Common Training Parameters ------------------------------------------------\ntraining:\n  use_wandb                   : True   # Toggle wandb logging\n  wandb_run_id                : "none" # "none" or "run_id"\n  resume                      : False  # Resume an existing run (run_id != \'none\')\n  epochs                      : 60\n  gradient_accumulation_steps : 1\n  wandb_project               : "HW4P1" # wandb project to log to\n\n###### Loss ----------------------------------------------------------------------\nloss: # Just good ol\' CrossEntropy\n  label_smoothing: 0.1\n\n###### Optimizer -----------------------------------------------------------------\noptimizer:\n  name: "adam" # Options: sgd, adam, adamw\n  lr: 5.0e-4   # Base learning rate\n\n  # Common parameters\n  weight_decay: 0.0001\n\n  # Parameter groups\n  param_groups:\n    - name: self_attn\n      patterns: []  # Will match all parameters containing keywords set their learning rate to 0.0001\n      lr: 0.0001    # LR for self_attn\n      layer_decay:\n        enabled: False\n        decay_rate: 0.8\n\n    - name: ffn\n      patterns: [] # Will match all parameters containing "ffn" and set their learning rate to 0.0001\n      lr: 0.0001   # LR for ffn\n      layer_decay:\n        enabled: False\n        decay_rate: 0.8\n\n  # Layer-wise learning rates\n  layer_decay:\n    enabled: False\n    decay_rate: 0.75\n\n  # SGD specific parameters\n  sgd:\n    momentum: 0.9\n    nesterov: True\n    dampening: 0\n\n  # Adam specific parameters\n  adam:\n    betas: [0.9, 0.999]\n    eps: 1.0e-8\n    amsgrad: False\n\n  # AdamW specific parameters\n  adamw:\n    betas: [0.9, 0.999]\n    eps: 1.0e-8\n    amsgrad: False\n\n###### Scheduler -----------------------------------------------------------------\nscheduler:\n  name: "cosine"  # Options: reduce_lr, cosine, cosine_warm\n\n  # ReduceLROnPlateau specific parameters\n  reduce_lr:\n    mode: "min"  # Options: min, max\n    factor: 0.1  # Factor to reduce learning rate by\n    patience: 10  # Number of epochs with no improvement after which LR will be reduced\n    threshold: 0.0001  # Threshold for measuring the new optimum\n    threshold_mode: "rel"  # Options: rel, abs\n    cooldown: 0  # Number of epochs to wait before resuming normal operation\n    min_lr: 0.0000001  # Minimum learning rate\n    eps: 1.0e-8  # Minimal decay applied to lr\n\n  # CosineAnnealingLR specific parameters\n  cosine:\n    T_max: 55  # Maximum number of iterations\n    eta_min: 1.0e-8  # Minimum learning rate\n    last_epoch: -1\n\n  # CosineAnnealingWarmRestarts specific parameters\n  cosine_warm:\n    T_0: 4  # Number of iterations for the first restart\n    T_mult: 4  # Factor increasing T_i after each restart\n    eta_min: 0.0000001  # Minimum learning rate\n    last_epoch: -1\n\n  # Warmup parameters (can be used with any scheduler)\n  warmup:\n    enabled: True\n    type: "exponential"  # Options: linear, exponential\n    epochs: 5\n    start_factor: 0.1\n    end_factor: 1.0\n')


# In[3]:


with open('config_lm.yaml', 'r') as file:
    config = yaml.safe_load(file)


# ## Tokenizer

# In[4]:


Tokenizer = H4Tokenizer(
    token_map  = config['tokenization']['token_map'],
    token_type = config['tokenization']['token_type']
)


# ## Datasets

# In[5]:


train_dataset  = LMDataset(
    partition  = config['data']['train_partition'],
    config     = config['data'],
    tokenizer  = Tokenizer
)

val_dataset    = LMDataset(
    partition  = config['data']['val_partition'],
    config     = config['data'],
    tokenizer  = Tokenizer
)

test_dataset   = LMDataset(
    partition  = config['data']['test_partition'],
    config     = config['data'],
    tokenizer  = Tokenizer
)

gc.collect()


# ## Dataloaders

# In[6]:


train_loader    = DataLoader(
    dataset     = train_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = True,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)

val_loader      = DataLoader(
    dataset     = val_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = False,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = val_dataset.collate_fn
)

test_loader     = DataLoader(
    dataset     = test_dataset,
    batch_size  = config['data']['batch_size'],
    shuffle     = False,
    num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)


# ### Dataloader Verification

# In[7]:


verify_dataloader(train_loader)


# In[8]:


verify_dataloader(val_loader)


# In[9]:


verify_dataloader(test_loader)


# ## Calculate Max Transcript Length
# 
# 
# 

# Calculating the maximum transcript length across your dataset is a crucial step when working with certain transformer models.
# -  We'll use sinusoidal positional encodings that must be precomputed up to a fixed maximum length.
# - This maximum length is a hyperparameter that determines:
#   - How long of a sequence your model can process
#   - The size of your positional encoding matrix
#   - Memory requirements during training and inference
# - `Requirements`: For this assignment, ensure your positional encodings can accommodate at least the longest sequence in your dataset to prevent truncation. However, you can set this value higher if you anticipate using your language model to work with longer sequences in future tasks (hint: this might be useful for P2! 😉).

# In[10]:


max_transcript_length = max(train_dataset.text_max_len, val_dataset.text_max_len, test_dataset.text_max_len)
print("="*50)
print(f"{'Global Max Transcript Length':<30} : {max_transcript_length}")
print("="*50)


# ## Model

# In[11]:


model_config = config['model']
model_config.update({
    'max_len': max_transcript_length,
    'num_classes': Tokenizer.vocab_size
})
model = DecoderOnlyTransformer(**model_config)

# Get some inputs from the text loader
for batch in train_loader:
    shifted_transcripts, golden_transcripts, transcript_lengths = batch
    print("Shape of shifted_transcripts : ", shifted_transcripts.shape)
    print("Shape of golden_transcripts  : ", golden_transcripts.shape)
    print("Shape of transcript_lengths  : ", transcript_lengths.shape)
    break

model_stats = summary(model, input_data=[shifted_transcripts, transcript_lengths])
print(model_stats)


# ## Wandb

# In[12]:


wandb.login(key="f83d7e9581d63eb876ad17665cf7c5c03d563770")


# ## Trainer
# 
# Every time you run the trainer, it will create a new directory in the `expts` folder with the following structure:
# ```
# expts/
#     └── {run_name}/
#         ├── config.yaml
#         ├── model_arch.txt
#         ├── checkpoints/
#         │   ├── checkpoint-best-metric-model.pth
#         │   └── checkpoint-last-epoch-model.pth
#         ├── attn/
#         │   └── {attention visualizations}
#         └── text/
#             └── {generated text outputs}
# ```

# In[13]:


trainer = LMTrainer(
    model=model,
    tokenizer=Tokenizer,
    config=config,
    run_name="lm-pretrain",
    config_file="config_lm.yaml",
    device=device
)


# ### Setup Optimizer and Scheduler
# 
# You can set your own optimizer and scheduler by setting the class members in the `LMTrainer` class.
# Eg:
# ```python
# trainer.optimizer = optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
# trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=config['training']['epochs'])
# ```
# 
# We also provide a utility function to create your own optimizer and scheduler with the congig and some extra bells and whistles. You are free to use it or not. Do read their code and documentation to understand how it works (`hw4lib/utils/*`).
# 

# #### Setting up the optimizer

# In[14]:


trainer.optimizer = create_optimizer(
    model=model,
    opt_config=config['optimizer']
)


# #### Creating a test scheduler and plotting the learning rate schedule

# In[15]:


test_scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

plot_lr_schedule(
    scheduler=test_scheduler,
    num_epochs=config['training']['epochs'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)


# #### Setting up the scheduler

# In[16]:


trainer.scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)


# # Train
# - Set your epochs

# In[17]:


trainer.train(train_loader, val_loader, epochs=config['training']['epochs'])


# # Evaluate
# 

# In[ ]:


test_metrics, test_generation_results = trainer.evaluate(test_loader)
# Cleanup
trainer.cleanup()


# # Submission
# To submit your assignment, you will need to create a `handin.tar` with the following directory structure:
# 
# ```
# handin/
# ├── mytorch/                     # Your implemented modules
# ├── test_metrics.json            # Results from evaluation
# ├── test_generated_results.json  # Sample text generations
# └── model_arch.txt               # Model architecture summary
# ```
# 
# - Simply run the cell below once you are satisfied with your current state and this will create the `handin.tar` file.
# - After running the above cell, you should see the handin.tar file in the current directory
# - Upload the `handin.tar` file to the `HW4P1` assignment on Autolab.

# # Create temporary handin directory
# if os.path.exists('handin'):
#     shutil.rmtree('handin')
# os.makedirs('handin')
# 
# # Copy mytorch directory
# shutil.copytree('mytorch', 'handin/mytorch')
# 
# # Save final results
# with open('handin/test_metrics.json', 'w') as f:
#     json.dump(test_metrics, f, indent=4)
# 
# with open('handin/test_generated_results.json', 'w') as f:
#     json.dump(test_generation_results['greedy'], f, indent=4)
# 
# # Save model architecture
# with open('handin/model_arch.txt', 'w') as f:
#     f.write(str(model_stats))
# 
# # Create tar file with all exclusions handled by filter
# with tarfile.open('handin.tar', 'w') as tar:
#     def filter_files(tarinfo):
#         # Skip unwanted files
#         if any(pattern in tarinfo.name for pattern in [
#             '.DS_Store',
#             '__pycache__',
#             '.pyc'
#         ]):
#             return None
#         return tarinfo
# 
#     tar.add('handin', arcname='handin', filter=filter_files)
# 
# # Cleanup
# shutil.rmtree('handin')
# 
# print("Created handin.tar successfully!")
# 
# ## After running the above cell, you should see the handin.tar file in the current directory
# !ls
