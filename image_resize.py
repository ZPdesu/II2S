from glob import glob
import numpy as np
import shutil
import os
import random
import PIL.Image
from pathlib import Path


folder_type = ['runs']
resize_type = ['Bicubic', 'LANCZOS']
resize_dic = {'Bicubic': PIL.Image.BICUBIC, 'LANCZOS':PIL.Image.LANCZOS}


for f_type in folder_type:
    for r_type in resize_type:
        img_names = glob(os.path.join(f_type, '*.png'))
        out_path = Path(os.path.join('Resized', f_type, r_type))
        out_path.mkdir(parents=True, exist_ok=True)

        for i in img_names:
            img = PIL.Image.open(i).convert('RGB')
            img = img.resize((256, 256), resize_dic[r_type])
            img.save(os.path.join(out_path, os.path.basename(i)))
