import os
import pickle

from imagededup.methods import PHash

data_dir = "/root/histology_lib/data/processed_images/"

phasher = PHash()
res = {}
for path in os.listdir(data_dir):
    encodings = phasher.encode_images(image_dir=f"{data_dir}/{path}")
    res = {**res, **encodings}

with open("phash.pkl", "wb") as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
