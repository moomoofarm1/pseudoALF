# config_pipe.py
import os
from pathlib import Path

HG_TOKEN = "YOUR_TOKEN" #"YOUR_TOKEN" 

# Default folders relative to where you launch (e.g., notebook root)
audio_dir = str(Path("rawdat").resolve())
pipe_dir  = str(Path("pipev4").resolve())
out_dir   = str(Path("out_condaspeechpipe").resolve())  # where clip*_16khz.flac may already exist

# Default log file should be in the NOTEBOOK folder (cwd), not necessarily out_dir
log_file = str((Path.cwd() / "audio_processing.log").resolve())

env = os.environ.copy()
env["AUDIOFOLDER"] = audio_dir
env["PIPEFOLDER"]  = pipe_dir
env["OUTFOLDER"]   = out_dir
env["LOG_FILE"]    = log_file
# Hugging Face authentication
if HG_TOKEN and HG_TOKEN != "YOUR_TOKEN":
    env["HF_TOKEN"] = HG_TOKEN
    env["HUGGINGFACE_HUB_TOKEN"] = HG_TOKEN  # backward / forward safe


# import os
# import Path
# from pathlib import Path

# HG_TOKEN = "YOUR_TOKEN"
# audio_dir = str(Path("rawdat").resolve())
# pipe_dir  = str(Path("pipev2").resolve()) 
# out_dir  = str(Path("out_condaspeechpipe").resolve())  # where clip*_16khz.flac already exist


# env = os.environ.copy()
# env["AUDIOFOLDER"] = audio_dir
# env["PIPEFOLDER"] = pipe_dir
# env["OUTFOLDER"] = out_dir
# env["LOG_FILE"]  = str(Path(#out_dir,
#                             "audio_processing.log").resolve())