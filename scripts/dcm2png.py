import os
import cv2
import pydicom
from pathlib import Path

from pydicom.pixel_data_handlers import apply_voi_lut
from joblib import Parallel, delayed
import numpy as np

all_files = list(Path("/mnt/vmk/datasets/tumor_mri").glob("**/*.dcm"))

outdir = "/root/histology_lib/data/processed_images/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

IMG_SIZE = 512


def convert_images(filename):
    try:
        if not os.path.exists(f"{outdir}/{filename.parts[5]}"):
            os.mkdir(f"{outdir}/{filename.parts[5]}")

        name = "_".join(filename.parts[6:])

        dicom = pydicom.dcmread(str(filename))
        img = apply_voi_lut(dicom.pixel_array, dicom)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        img = (img - img.min()) / float(img.max() - img.min())
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # denoising of image saving it into dst image
        img = cv2.fastNlMeansDenoising(img, h=4)

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        cv2.imwrite(f"{outdir}/{filename.parts[5]}/{name.replace('.dcm','.png')}", img)
    except:
        print(f"{'/'.join(filename.parts[5:])}")


Parallel(n_jobs=200)(delayed(convert_images)(i) for i in all_files)
