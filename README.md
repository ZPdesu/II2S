# II2S

1. cd II2S
2. download [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1rKxxPm4FHnna1E-D5fBsj5gjG0sWnodr/view?usp=sharing)
3. python run.py --In_W_space -steps 1300  -learning_rate 0.01 --input_dir input --output_dir output/generated_images --NPY_folder output/latent_representations --L2_regularizer --l2_regularize_weight 0.001
