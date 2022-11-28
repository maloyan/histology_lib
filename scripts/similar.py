import difflib
import os
import pickle
from pathlib import Path
import distance
from imagededup.methods import PHash
from tqdm import tqdm

data_dir = "/root/histology_lib/data/processed_images/"
all_files = list(Path(data_dir).glob("**/*.png"))

phasher = PHash()

with open("/root/histology_lib/scripts/phash.pkl", "rb") as f:
    d = pickle.load(f)

query = d["Head - 10978_Ax_T1_FSE_4_IM-0002-0016.png"]

res = {}
for k, v in tqdm(d.items(), total=len(d)):
    res[k] = distance.hamming(query, v) #difflib.SequenceMatcher(None, query, v).ratio()


filtered_res = {}

# hash is 16

threshold = 13

for i in tqdm(all_files, total=len(all_files)):
    if (
        i.parts[-1] in res
        and res[i.parts[-1]] <= threshold
        and ("T1" in i.parts[-1] or "T2" in i.parts[-1])
    ):
        filtered_res["/".join(i.parts[5:])] = res[i.parts[-1]]


filtered_res = dict(
    sorted(filtered_res.items(), key=lambda item: item[1], reverse=True)
)

with open("similar.pkl", "wb") as f:
    pickle.dump(filtered_res, f, pickle.HIGHEST_PROTOCOL)
