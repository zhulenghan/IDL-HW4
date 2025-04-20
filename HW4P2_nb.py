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
# os.environ['GITHUB_TOKEN'] = "your_github_token_here"
# 
# GITHUB_USERNAME = "your_github_username_here"
# REPO_NAME       = "your_github_repo_name_here"
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
# os.environ['GITHUB_TOKEN'] = "your_github_token_here"
# 
# GITHUB_USERNAME = "your_github_username_here"
# REPO_NAME       = "your_github_repo_name_here"
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
# os.environ['GITHUB_TOKEN'] = "your_github_token_here"
# 
# GITHUB_USERNAME = "your_github_username_here"
# REPO_NAME       = "your_github_repo_name_here"
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
# - If your setup was done correctly, you should be able to run the following cell without any issues.

# In[1]:


from hw4lib.data import (
    H4Tokenizer,
    ASRDataset,
    verify_dataloader
)
from hw4lib.model import (
    DecoderOnlyTransformer,
    EncoderDecoderTransformer
)
from hw4lib.utils import (
    create_scheduler,
    create_optimizer,
    plot_lr_schedule
)
from hw4lib.trainers import ( 
    ASRTrainer,
    ProgressiveTrainer
)
from torch.utils.data import DataLoader
import yaml
import gc
import torch
from torchinfo import summary
import os
import json
import wandb
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# # Implementations
# - `NOTE`: All of these implementations have detailed specification, implementation details, and hints in their respective source files. Make sure to read all of them in their entirety to understand the implementation details!

# ## Dataset Implementation
# - Implement the `ASRDataset` class in `hw4lib/data/asr_dataset.py`.
# - You will have to implement parts of `__init__` and completely implement the `__len__`, `__getitem__` and `collate_fn` methods.
# - Run the cell below to check your implementation.
# 

# !python -m tests.test_dataset_asr

# ## Model Implementations
# 
# Overview:
# 
# - Implement the `CrossAttentionLayer` class in `hw4lib/model/sublayers.py`.
# - Implement the `CrossAttentionDecoderLayer` class in `hw4lib/model/decoder_layers.py`.
# - Implement the `SelfAttentionEncoderLayer` class in `hw4lib/model/encoder_layers.py`. This will be mostly a copy-paste of the `SelfAttentionDecoderLayer` class in `hw4lib/model/decoder_layers.py` with one minor diffrence: it can attend to all positions in the input sequence.
# - Implement the `EncoderDecoderTransformer` class in `hw4lib/model/transformers.py`.

# ### Transformer Sublayers
# - Now, Implement the `CrossAttentionLayer` class in `hw4lib/model/sublayers.py`.
# - `NOTE`: You should have already implemented the `SelfAttentionLayer`, and `FeedForwardLayer` classes in `hw4lib/model/sublayers.py`.
# - Run the cell below to check your implementation.

# !python -m tests.test_sublayer_crossattention

# ### Transformer Cross-Attention Decoder Layer
# - Implement the `CrossAttentionDecoderLayer` class in `hw4lib/model/decoder_layers.py`.
# - Then run the cell below to check your implementation.
# 

# !python -m tests.test_decoderlayer_crossattention

# ### Transformer Self-Attention Encoder Layer
# - Implement the `SelfAttentionEncoderLayer` class in `hw4lib/model/encoder_layers.py`.
# - Then run the cell below to check your implementation.
# 
# 
# 

# !python -m tests.test_encoderlayer_selfattention

# ### Encoder-Decoder Transformer
# 
# - Implement the  `EncoderDecoderTransformer` class in `hw4lib/model/transformers.py`.
# - Then run the cell below to check your implementation.

# !python -m tests.test_transformer_encoder_decoder

# ## Decoding Implementation
# - We highly recommend you to implement the `generate_beam` method of the `SequenceGenerator` class in `hw4lib/decoding/sequence_generator.py`.
# - Then run the cell below to check your implementation.
# - `NOTE`: This is an optional but highly recommended task for `HW4P2` to ease the journey to high cutoffs!

# !python -m tests.test_decoding --mode greedy

# ## Trainer Implementation
# You will have to do some minor in-filling for the `ASRTrainer` class in `hw4lib/trainers/asr_trainer.py` before you can use it.
# - Fill in the `TODO`s in the `__init__`.
# - Fill in the `TODO`s in the `_train_epoch`.
# - Fill in the `TODO`s in the `recognize` method.
# - Fill in the `TODO`s in the `_validate_epoch`.
# - Fill in the `TODO`s in the `train` method.
# - Fill in the `TODO`s in the `evaluate` method.
# 
# `WARNING`: There are no test's for this. Implement carefully!

# # Experiments
# From this point onwards you may want to switch to a `GPU` runtime.
# - `OBJECTIVE`: Optimize your model for `CER` on the test set.

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

# In[6]:


#get_ipython().run_cell_magic('writefile', 'config.yaml', '\nName                      : "lenghanz"\n\n###### Tokenization ------------------------------------------------------------\ntokenization:\n  token_type                : "10k"       # [char, 1k, 5k, 10k]\n  token_map :\n      \'char\': \'hw4lib/data/tokenizer_jsons/tokenizer_char.json\'\n      \'1k\'  : \'hw4lib/data/tokenizer_jsons/tokenizer_1000.json\'\n      \'5k\'  : \'hw4lib/data/tokenizer_jsons/tokenizer_5000.json\'\n      \'10k\' : \'hw4lib/data/tokenizer_jsons/tokenizer_10000.json\'\n\n###### Dataset -----------------------------------------------------------------\ndata:\n  root                 : "hw4_data/hw4p2_data"  # TODO: Set the root path of your data\n  train_partition      : "train-clean-100"  # paired text-speech for ASR pre-training\n  val_partition        : "dev-clean"        # paired text-speech for ASR pre-training\n  test_partition       : "test-clean"       # paired text-speech for ASR pre-training\n  subset               : 1.0                # Load a subset of the data (for debugging, testing, etc\n  batch_size           : 32            #\n  NUM_WORKERS          : 32            # Set to 0 for CPU\n  norm                 : \'global_mvn\' # [\'global_mvn\', \'cepstral\', \'none\']\n  num_feats            : 80\n\n  ###### SpecAugment ---------------------------------------------------------------\n  specaug                   : True  # Set to True if you want to use SpecAugment\n  specaug_conf:\n    apply_freq_mask         : True\n    freq_mask_width_range   : 7\n    num_freq_mask           : 4\n    apply_time_mask         : True\n    time_mask_width_range   : 50\n    num_time_mask           : 4\n\n###### Network Specs -------------------------------------------------------------\nmodel: # Encoder-Decoder Transformer (HW4P2)\n  # Speech embedding parameters\n  input_dim: 80              # Speech feature dimension\n  time_reduction: 2          # Time dimension downsampling factor\n  reduction_method: \'lstm\'   # The source_embedding reduction method [\'lstm\', \'conv\', \'both\']\n\n  # Architecture parameters\n  d_model: 384           # Model dimension\n  num_encoder_layers: 10  # Number of encoder layers\n  num_decoder_layers: 3  # Number of decoder layers\n  num_encoder_heads: 6   # Number of encoder attention heads\n  num_decoder_heads: 4   # Number of decoder attention heads\n  d_ff_encoder: 768     # Feed-forward dimension for encoder\n  d_ff_decoder: 672     # Feed-forward dimension for decoder\n  skip_encoder_pe: False # Whether to skip positional encoding for encoder\n  skip_decoder_pe: False # Whether to skip positional encoding for decoder\n\n  # Common parameters\n  dropout: 0.1          # Dropout rate\n  layer_drop_rate: 0.1  # Layer dropout rate\n  weight_tying: False   # Whether to use weight tying\n\n###### Common Training Parameters ------------------------------------------------\ntraining:\n  use_wandb                   : True   # Toggle wandb logging\n  wandb_run_id                : "" # "none" or "run_id"\n  resume                      : True   # Resume an existing run (run_id != \'none\')\n  gradient_accumulation_steps : 2\n  wandb_project               : "HW4P2" # wandb project to log to\n\n###### Loss ----------------------------------------------------------------------\nloss: # Just good ol\' CrossEntropy\n  label_smoothing: 0.1\n  ctc_weight: 0.2\n\n###### Optimizer -----------------------------------------------------------------\noptimizer:\n  name: "adamw" # Options: sgd, adam, adamw\n  lr: 0.0004    # Base learning rate\n\n  # Common parameters\n  weight_decay: 0.000001\n\n  # Parameter groups\n  # You can add more param groups as you want and set their learning rates and patterns\n  param_groups:\n    # - name: self_attn\n    #   patterns: [\'dec_layers.0.self_attn\', \'dec_layers.1.self_attn\', \'dec_layers.2.self_attn\', \'dec_layers.3.self_attn\']  # Will match all parameters containing "ffn" and set their learning rate to 0.0002\n    #   lr: 0.00004    # LR for self_attn\n    #   layer_decay:\n    #     enabled: False\n    #     decay_rate: 0.8\n\n    # - name: ffn\n    #   patterns: [\'dec_layers.0.ffn\', \'dec_layers.1.ffn\', \'dec_layers.2.ffn\', \'dec_layers.3.ffn\'] # Will match all parameters containing "ffn" and set their learning rate to 0.0002\n    #   lr: 0.00004   # LR for ffn\n    #   layer_decay:\n    #     enabled: False\n    #     decay_rate: 0.8\n\n\n  # Layer-wise learning rates\n  layer_decay:\n    enabled: False\n    decay_rate: 0.75\n\n  # SGD specific parameters\n  sgd:\n    momentum: 0.9\n    nesterov: True\n    dampening: 0\n\n  # Adam specific parameters\n  adam:\n    betas: [0.9, 0.999]\n    eps: 1.0e-8\n    amsgrad: False\n\n  # AdamW specific parameters\n  adamw:\n    betas: [0.9, 0.999]\n    eps: 1.0e-8\n    amsgrad: False\n\n###### Scheduler -----------------------------------------------------------------\nscheduler:\n  name: "cosine"  # Options: reduce_lr, cosine, cosine_warm\n\n  # ReduceLROnPlateau specific parameters\n  reduce_lr:\n    mode: "min"  # Options: min, max\n    factor: 0.1  # Factor to reduce learning rate by\n    patience: 10  # Number of epochs with no improvement after which LR will be reduced\n    threshold: 0.0001  # Threshold for measuring the new optimum\n    threshold_mode: "rel"  # Options: rel, abs\n    cooldown: 0  # Number of epochs to wait before resuming normal operation\n    min_lr: 0.0000001  # Minimum learning rate\n    eps: 1e-8  # Minimal decay applied to lr\n\n  # CosineAnnealingLR specific parameters\n  cosine:\n    T_max: 58  # Maximum number of iterations\n    eta_min: 0.00001  # Minimum learning rate\n    last_epoch: -1\n\n  # CosineAnnealingWarmRestarts specific parameters\n  cosine_warm:\n    T_0: 10    # Number of iterations for the first restart\n    T_mult: 10 # Factor increasing T_i after each restart\n    eta_min: 0.0000001  # Minimum learning rate\n    last_epoch: -1\n\n  # Warmup parameters (can be used with any scheduler)\n  warmup:\n    enabled: True\n    type: "exponential"  # Options: linear, exponential\n    epochs: 2\n    start_factor: 0.1\n    end_factor: 1.0\n')


# In[5]:


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# ## Tokenizer

# In[4]:


Tokenizer = H4Tokenizer(
    token_map  = config['tokenization']['token_map'],
    token_type = config['tokenization']['token_type']
)


# ## Datasets

# In[5]:


train_dataset = ASRDataset(
    partition=config['data']['train_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=True,
    global_stats=None  # Will compute stats from training data
)

# TODO: Get the computed global stats from training set
global_stats = None
if config['data']['norm'] == 'global_mvn':
    global_stats = (train_dataset.global_mean, train_dataset.global_std)
    print(f"Global stats computed from training set.")

val_dataset = ASRDataset(
    partition=config['data']['val_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

test_dataset = ASRDataset(
    partition=config['data']['test_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
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

gc.collect()


# ### Dataloader Verification

# In[7]:


verify_dataloader(train_loader)


# In[8]:


verify_dataloader(val_loader)


# In[9]:


verify_dataloader(test_loader)


# ## Calculate Max Lengths
# Calculating the maximum transcript length across your dataset is a crucial step when working with certain transformer models.
# -  We'll use sinusoidal positional encodings that must be precomputed up to a fixed maximum length.
# - This maximum length is a hyperparameter that determines:
#   - How long of a sequence your model can process
#   - The size of your positional encoding matrix
#   - Memory requirements during training and inference
# - `Requirements`: For this assignment, ensure your positional encodings can accommodate at least the longest sequence in your dataset to prevent truncation. However, you can set this value higher if you anticipate using your languagemodel to work with longer sequences in future tasks (hint: this might be useful for P2! 😉).
# - `NOTE`: We'll be using the same positional encoding matrix for all sequences in your dataset. Take this into account when setting your maximum length.

# In[10]:


max_feat_len       = max(train_dataset.feat_max_len, val_dataset.feat_max_len, test_dataset.feat_max_len)
max_transcript_len = max(train_dataset.text_max_len, val_dataset.text_max_len, test_dataset.text_max_len)
max_len            = max(max_feat_len, max_transcript_len)

print("="*50)
print(f"{'Max Feature Length':<30} : {max_feat_len}")
print(f"{'Max Transcript Length':<30} : {max_transcript_len}")
print(f"{'Overall Max Length':<30} : {max_len}")
print("="*50)


# ## Wandb

# In[11]:


wandb.login(key="f83d7e9581d63eb876ad17665cf7c5c03d563770")


# ## Training
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
# 

# ### Training Strategy 1: Cold-Start Trainer

# #### Model Load (Default)

# In[12]:


model_config = config['model'].copy()
model_config.update({
    'max_len': max_len,
    'num_classes': Tokenizer.vocab_size
})

model = EncoderDecoderTransformer(**model_config)

# Get some inputs from the train dataloader
for batch in train_loader:
    padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths = batch
    break


model_stats = summary(model, input_data=[padded_feats, padded_shifted, feat_lengths, transcript_lengths])
print(model_stats)


# checkpoint_path = "/root/autodl-tmp/IDL-HW4/expts/pretrain-384-1536-4-4-8-8/checkpoints/checkpoint-last-epoch-model.pth"
# model.load_state_dict(torch.load(checkpoint_path))

# In[13]:


for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")


# #### Initialize Trainer
# 
# If you need to reload the model from a checkpoint, you can do so by calling the `load_checkpoint` method.
# 
# ```python
# checkpoint_path = "path/to/checkpoint.pth"
# trainer.load_checkpoint(checkpoint_path)
# ```
# 

# checkpoint_path = "/root/autodl-tmp/IDL-HW4/expts/pretrain-384-1536-4-4-8-8/checkpoints/checkpoint-last-epoch-model.pth"
# trainer.load_checkpoint(checkpoint_path)

# In[ ]:


trainer = ASRTrainer(
    model=model,
    tokenizer=Tokenizer,
    config=config,
    run_name="pretrain-384-768-dropout0.3-120epoch",
    config_file="config.yaml",
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

# In[15]:


trainer.optimizer = create_optimizer(
    model=model,
    opt_config=config['optimizer']
)


# #### Creating a test scheduler and plotting the learning rate schedule

# In[16]:


test_scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

plot_lr_schedule(
    scheduler=test_scheduler,
    num_epochs=60,
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    max_groups=10,
)


# #### Setting up the scheduler

# In[17]:


trainer.scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)


# #### Train
# - Set your epochs

# In[ ]:


trainer.train(train_loader, val_loader, epochs=60)


# #### Inference
# 

# In[23]:


# Define the recognition config: Greedy search
recognition_config = {
    'num_batches': None,
    'temperature': 1.0,
    'repeat_penalty': 1.0,
    'lm_weight': None,
    'lm_model': None,
    'beam_width': 2, # Beam width of 1 reverts to greedy
}

# Recognize with the shallow fusion config
config_name = "test"
print(f"Evaluating with {config_name} config")
results = trainer.recognize(test_loader, recognition_config, config_name=config_name, max_length=max_transcript_len)


# Calculate metrics on full batch
generated = [r['generated'] for r in results]
results_df = pd.DataFrame(
    {
        'id': range(len(generated)),
        'transcription': generated
    }
)

# Cleanup (Will end wandb run)
trainer.cleanup()


# ## Submit to Kaggle

# ### Authenticate Kaggle
# In order to use the Kaggle’s public API, you must first authenticate using an API token. Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
# - `TODO`: Set your kaggle username and api key here based on the API credentials listed in the kaggle.json
# 
# 
# 

# In[27]:


import os
os.environ["KAGGLE_USERNAME"] = "dean8211"
os.environ["KAGGLE_KEY"] = "93c897dd68c2f58545c6e76d2097609f"


# In[28]:


results_df.head()


# ### Submit

# In[ ]:


results_df.to_csv("results.csv", index=False)
#get_ipython().system('kaggle competitions submit -c 11785-s25-hw4p2-asr -f results.csv -m "My Submission"')
os.system('kaggle competitions submit -c 11785-s25-hw4p2-asr -f results.csv -m "My Submission"')
