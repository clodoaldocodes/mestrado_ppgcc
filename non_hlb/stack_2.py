import os
import numpy as np
import rasterio 
from PIL import Image
import cv2
import math
from alive_progress import alive_bar

path = "C:/Users/cdsfj/Desktop/DOCUMENTOS/non_HLB"
output_stack = path + "/stack/"
format_use = "TIFF"
use_all_bands = False
use_q = "Q4"#"Q6"

def apply_contrast_and_gamma(band, contrast_factor, gamma):
    # Apply contrast
    min_value = np.min(band)
    max_value = np.max(band)
    #contrasted_band = (band - min_value) * contrast_factor + min_value
    contrasted_band = band
    
    # Apply gamma correction
    val_gamma = np.power(contrasted_band, gamma).clip(0, 255).astype(np.uint8)
    
    return val_gamma

folders_path = []
if use_all_bands:
    bands = []
else:
    if use_q == "Q4":
        bands = ["650", "550", "430"]
    elif use_q == "Q6":
        bands = ["685", "560", "450"]
    else:
        bands = ["650"]

if not os.path.exists(output_stack):
    os.makedirs(output_stack)

if len(bands) > 1:
    output_png = path + "/rgb_" + use_q.lower() + "/"
else:
    output_png=  path + "/gray_" + use_q.lower() + "/"

if not os.path.exists(output_png):
    os.makedirs(output_png)

for folder_path_1, folders, files in os.walk(path):
    folders_path.append(folder_path_1)
    if use_all_bands:
        bands.append(folder_path_1.split("\\")[-1])

k = 2
names_to_find_complete = [f for f in os.listdir(folders_path[k]) if os.path.isfile(os.path.join(folders_path[k], f))]
names_to_find = [name.split("_")[0] for name in names_to_find_complete]

list_tiffs = []
for i in range(np.size(names_to_find, axis=0)):
    file_to_load = []
    for j in range(np.size(bands, axis=0)):
        file_to_load.append(path + "/" + str(bands[j]) + "NM/" + names_to_find[i] + "_" + str(bands[j]) + "nm." + format_use)
    
    with rasterio.open(file_to_load[0]) as src0:
        meta = src0.meta
    
    meta.update(count=len(file_to_load))
    
    k = 0
    filename_actual = names_to_find[i] + "_stack." + format_use 
    path_filename_output = output_stack + filename_actual
    list_tiffs.append(path_filename_output)

    with rasterio.open(path_filename_output, "w", **meta) as dst:
        with alive_bar(len(file_to_load), title=f'Processing {names_to_find[i]}') as bar:
            for id, layer in enumerate(file_to_load, start=1):
                with rasterio.open(layer) as src1:
                    band = src1.read(1)
                    contrast_factor = 1
                    mid = 0.5
                    mean = np.mean(band)
                    gamma = math.log(mid*255)/math.log(mean)
                    gamma = 1
                    corrected_band = apply_contrast_and_gamma(band, contrast_factor, gamma)
                    dst.write_band(id, corrected_band)
                    bar()
                          
for i in range(np.size(list_tiffs)):
    with alive_bar(title=f'Processing {list_tiffs[i]}') as bar:     
        with Image.open(list_tiffs[i]) as img:
            filename = output_png + names_to_find[i] + "_stack.png" 
            img.save(filename, format="PNG")
            bar()
