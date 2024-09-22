import os, sys
os.system("pip install pyworld") # ==0.3.3

now_dir = os.getcwd()
sys.path.append(now_dir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

# Download models
shell_script = './tools/dlmodels.sh'
os.system(f'chmod +x {shell_script}')
os.system('apt install git-lfs')
os.system('git lfs install')
os.system('apt-get -y install aria2')
os.system('aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d . -o hubert_base.pt')
try:
    return_code = os.system(shell_script)
    if return_code == 0:
        print("Shell script executed successfully.")
    else:
        print(f"Shell script failed with return code {return_code}")
except Exception as e:
    print(f"An error occurred: {e}")


import logging
import shutil
import threading
import lib.globals.globals as rvc_globals
from LazyImport import lazyload
import mdx
from mdx_processing_script import get_model_list,id_to_ptm,prepare_mdx,run_mdx
math = lazyload('math')
import traceback
import warnings
tensorlowest = lazyload('tensorlowest')
from random import shuffle
from subprocess import Popen
from time import sleep
import json
import pathlib

import fairseq
logging.getLogger("faiss").setLevel(logging.WARNING)
import faiss
gr = lazyload("gradio")
np = lazyload("numpy")
torch = lazyload('torch')
re = lazyload('regex')
SF = lazyload("soundfile")
SFWrite = SF.write
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
import datetime


from glob import glob1
import signal
from signal import SIGTERM
import librosa

from configs.config import Config
from i18n import I18nAuto
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
#from infer.modules.uvr5.modules import uvr
from infer.modules.vc.modules import VC
from infer.modules.vc.utils import *
from infer.modules.vc.pipeline import Pipeline
import lib.globals.globals as rvc_globals
math = lazyload('math')
ffmpeg = lazyload('ffmpeg')
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from bark import SAMPLE_RATE

import easy_infer
import audioEffects
from infer.lib.csvutil import CSVutil

from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_, _audio_pre_new
from MDXNet import MDXNetDereverb
from infer.lib.audio import load_audio


from sklearn.cluster import MiniBatchKMeans

import time
import csv

from shlex import quote as SQuote




RQuote = lambda val: SQuote(str(val))

tmp = os.path.join(now_dir, "TEMP")
runtime_dir = os.path.join(now_dir, "runtime/Lib/site-packages")
directories = ['logs', 'audios', 'datasets', 'weights', 'audio-others' , 'audio-outputs']

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)

os.makedirs(tmp, exist_ok=True)
for folder in directories:
    os.makedirs(os.path.join(now_dir, folder), exist_ok=True)


os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


if not os.path.isdir("csvdb/"):
    os.makedirs("csvdb")
    frmnt, stp = open("csvdb/formanting.csv", "w"), open("csvdb/stop.csv", "w")
    frmnt.close()
    stp.close()

global DoFormant, Quefrency, Timbre

try:
    DoFormant, Quefrency, Timbre = CSVutil("csvdb/formanting.csv", "r", "formanting")
    DoFormant = (
        lambda DoFormant: True
        if DoFormant.lower() == "true"
        else (False if DoFormant.lower() == "false" else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil("csvdb/formanting.csv", "w+", "formanting", DoFormant, Quefrency, Timbre)
