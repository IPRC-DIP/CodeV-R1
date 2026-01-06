# Supervised Fine-Tuning (SFT) with LLaMA-Factory

This section provides instructions for performing supervised fine-tuning (SFT) using the provided configuration file and dataset.

## 📁 Files Provided

- **Configuration File**: `codev_r1_distill.yaml`  
  This file contains the hyperparameters and settings required for the SFT process.

## 📊 Dataset

The training data should be downloaded from the following Hugging Face dataset repository:  
**https://huggingface.co/datasets/zhuyaoyu/CodeV-R1-dataset**

Specifically, use the file:  
**`codev_r1_sft.jsonl`**

## 🚀 How to Run

1. **Place the configuration file**  
   Copy `codev_r1_distill.yaml` into the `examples/train_full/` directory within your LLaMA-Factory project.

   Add the following content into `LLaMA-Factory/data/dataset_info.json`:
   
   ```json
     "codev_r1_sft": {
       "file_name": "your/path/to/codev_r1_sft.jsonl",
       "columns": {
         "system": "system",
         "prompt": "prompt",
         "response": "response"
       }
     },
   ```
   
   (The `file_name` should be changed to your real path.)
   
2. **Run the training command**  
   Navigate to the root directory of LLaMA-Factory and execute the following command:

   ```bash
   llamafactory-cli train examples/train_full/codev_r1_distill.yaml
   ```