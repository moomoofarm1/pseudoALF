import torch
import gc

print("Cleaning GPU memory...")

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # extra cleanup for inter-process memory

print("Done.")