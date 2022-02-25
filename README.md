# II2S

## Setup
### Requirements
- You must have a GPU with at least XXX vRAM (e.g. )
- You must have nvidia drivers installed (e.g. `nvidia-smi` lists your GPUs)
- All instructions assume a bash shell (e.g. WSL for windows users)
### Instructions
Setup instructions (you should be able to paste each line into a bash shell):
```bash
# Clone this repository
git clone git@github.com:ZPdesu/II2S.git 
cd II2S

# Install python dependancies
conda env create
conda activate ii2s


python run.py --input_dir input --output_dir output -dc experiment.yml

python run.py \
    --In_W_space \
    -steps 1300 \
    -learning_rate 0.01 \
    --input_dir input \
    --output_dir output/generated_images \
    --NPY_folder output/latent_representations \
    --L2_regularizer \
    --l2_regularize_weight 0.001
```
