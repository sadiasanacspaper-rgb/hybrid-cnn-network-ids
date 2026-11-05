
# ##################################################################################################
#  ADVANCED HYBRID NETWORK INTRUSION DETECTION SYSTEM
#  CIC-IDS2017  +  CSE-CIC-IDS2018   (combined / harmonized)
#
#  Hybrid architecture:  1D-CNN deep feature extractor  ->  { SVM | Decision Tree | Random Forest }
#  Extras: MITRE ATT&CK mapping, dataset-wise generalization, SOC-style Gradio dashboard.
#
#  >>> PASTE THIS ENTIRE FILE INTO ONE GOOGLE COLAB CELL AND RUN IT. <<<
#  It installs packages, downloads the datasets automatically (Hugging Face Hub, no login,
#  no manual upload), harmonizes them, trains the 3 hybrids, evaluates, visualizes,
#  saves artifacts and launches an interactive dashboard.
#
#  Research-integrity note: every number printed/plotted is computed at runtime from the
#  actual experiment. Nothing is hard-coded. If the datasets cannot be downloaded the code
#  prints the real error and (only then) falls back to a clearly-labelled SYNTHETIC dataset
#  so the pipeline can still be demonstrated - the report will say SYNTHETIC in that case.
# ##################################################################################################


# ============================================================
# 0. CONFIGURATION  (edit these, then run the whole cell)
# ============================================================
SEED = 42

# ---- memory / sampling controls -------------------------------------------------
USE_SAMPLING          = True      # True = Colab-friendly stratified subsample. False = use everything (needs lots of RAM).
MAX_ROWS_PER_DATASET  = 320_000   # hard cap of rows kept per dataset when USE_SAMPLING=True
MAX_ROWS_PER_CLASS    = 25_000    # per (attack class) cap  -> keeps minority classes, caps majority
BENIGN_CAP            = 90_000    # separate (larger) cap for the BENIGN class per dataset
CHUNK_SIZE            = 200_000   # rows per pandas chunk while streaming the big CSVs
DOWNLOAD_BIG_2018_FILE= False     # the 20-02-2018 file is ~4 GB; leave False unless USE_SAMPLING=False and RAM is huge

# ---- model controls -----------------------------------------------------------
CNN_EPOCHS            = 18
CNN_BATCH             = 512
CNN_EMBED_DIM         = 64        # size of the learned CNN embedding fed to SVM / DT / RF
TUNE_SUBSAMPLE        = 60_000    # rows used for hyper-parameter search (speed); best model refit on full train
SVM_RBF_MAX_TRAIN     = 15_000    # train <= this -> RBF SVM on full train; else RBF SVM on a 30k stratified subsample
ACCURACY_DEEP_DIVE    = True      # also train standalone classical models on RAW features (adds ~2-4 min)

# ---- dashboard --------------------------------------------------------------
LAUNCH_DASHBOARD      = True
GRADIO_SHARE          = True      # public share link (needed to view the UI from Colab)

OUTPUT_DIR            = "IDS_Results"


# ============================================================
# 1. INSTALLATION
# ============================================================
import sys, subprocess, importlib

def _pip(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

_need = {
    "huggingface_hub": "huggingface_hub>=0.23",
    "gradio":          "gradio>=4.0",
    "plotly":          "plotly>=5.18",
    "psutil":          "psutil",
    "seaborn":         "seaborn",
    "sklearn":         "scikit-learn>=1.3",
    "tensorflow":      "tensorflow",
    "pyarrow":         "pyarrow",
    "joblib":          "joblib",
}
_missing = []
for mod, spec in _need.items():
    try:
        importlib.import_module(mod)
    except Exception:
        _missing.append(spec)
if _missing:
    print("Installing:", _missing)
    _pip(_missing)
print("Installation step complete.")


# ============================================================
# 2. IMPORTS  +  REPRODUCIBILITY  +  GPU DETECTION
# ============================================================
import os, gc, glob, time, json, random, warnings, zipfile, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score, roc_curve,
                             precision_recall_curve, classification_report)

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

GPUS = tf.config.list_physical_devices("GPU")
for g in GPUS:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mem_report(tag=""):
    vm = psutil.virtual_memory()
    print(f"[MEM {tag}] used={vm.used/1e9:5.2f} GB  avail={vm.available/1e9:5.2f} GB  ({vm.percent:.0f}%)")

print("="*70)
print(f"Experiment seed: {SEED}")
print(f"TensorFlow: {tf.__version__}   GPU available: {bool(GPUS)}  ->  {[g.name for g in GPUS]}")
mem_report("startup")
print("="*70)


# ============================================================
# 3. DATASET ACQUISITION  (automatic, no manual upload)
# ============================================================
# Primary source  : Hugging Face Hub public dataset repos (no token required)
#   CIC-IDS2017     -> c01dsnap/CIC-IDS2017            (8 canonical MachineLearningCVE CSVs)
#   CSE-CIC-IDS2018 -> pcy12345BSU/CSE-CIC-IDS-2018    (10 canonical TrafficForML CSVs)
# Fallback source : abluva/CIC-IDS-2017-V2 , abluva/CSE-CIC-IDS2018-V2  (single normalized CSV each)
# Last resort     : clearly-labelled SYNTHETIC generator (only if all downloads fail)
from huggingface_hub import hf_hub_download

DATA_ROOT = "ids_data"
DIR_2017  = os.path.join(DATA_ROOT, "cicids2017")
DIR_2018  = os.path.join(DATA_ROOT, "csecicids2018")
os.makedirs(DIR_2017, exist_ok=True)
os.makedirs(DIR_2018, exist_ok=True)

FILES_2017 = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
FILES_2018 = [
    "data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",   # FTP/SSH brute force
    "data/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",    # DoS GoldenEye / Slowloris
    "data/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",      # DoS SlowHTTPTest / Hulk
    "data/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",   # DDoS LOIC-UDP / HOIC
    "data/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",    # Web brute force / XSS / SQLi
    "data/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",      # Web brute force / XSS / SQLi
    "data/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",   # Infiltration
    "data/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",    # Infiltration
    "data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",      # Bot
]
if DOWNLOAD_BIG_2018_FILE:
    FILES_2018.insert(3, "data/Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv")  # ~4 GB DDoS day

DATASET_IS_SYNTHETIC = False


def _try_download(repo_id, filename, local_dir):
    """Download one file from a HF dataset repo, skip if already present. Returns local path or None."""
    target = os.path.join(local_dir, os.path.basename(filename))
    if os.path.exists(target) and os.path.getsize(target) > 1000:
        return target
    try:
        p = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                            local_dir=local_dir)
        # flatten (hf keeps sub-dirs like data/...)
        if os.path.abspath(p) != os.path.abspath(target):
            try:
                os.replace(p, target)
            except Exception:
                target = p
        print(f"   downloaded  {os.path.basename(filename):55s} "
              f"({os.path.getsize(target)/1e6:7.1f} MB)")
        return target
    except Exception as e:
        print(f"   FAILED      {filename}  ->  {type(e).__name__}: {e}")
        return None


def download_datasets():
    """Acquire both datasets. Returns (list_of_2017_paths, list_of_2018_paths)."""
    global DATASET_IS_SYNTHETIC
    print("\n--- Acquiring CIC-IDS2017 ------------------------------------------")
    paths_2017 = []
    for f in FILES_2017:
        p = _try_download("c01dsnap/CIC-IDS2017", f, DIR_2017)
        if p:
            paths_2017.append(p)
    if not paths_2017:
        print("   primary source failed -> trying fallback abluva/CIC-IDS-2017-V2")
        p = _try_download("abluva/CIC-IDS-2017-V2", "CIC-IDS-2017-V2.csv", DIR_2017)
        if p:
            paths_2017.append(p)

    print("\n--- Acquiring CSE-CIC-IDS2018 --------------------------------------")
    paths_2018 = []
    for f in FILES_2018:
        p = _try_download("pcy12345BSU/CSE-CIC-IDS-2018", f, DIR_2018)
        if p:
            paths_2018.append(p)
    if not paths_2018:
        print("   primary source failed -> trying fallback abluva/CSE-CIC-IDS2018-V2")
        p = _try_download("abluva/CSE-CIC-IDS2018-V2", "CSE-CIC-IDS-2018-V2.csv", DIR_2018)
        if p:
            paths_2018.append(p)

    if not paths_2017 or not paths_2018:
        print("\n!!! One or both datasets could not be downloaded. "
              "Generating a clearly-labelled SYNTHETIC fallback so the pipeline still runs.")
        DATASET_IS_SYNTHETIC = True
        return _make_synthetic()
    return paths_2017, paths_2018


def _make_synthetic():
    """Synthetic CICFlowMeter-like data. ONLY used when real downloads fail. Clearly flagged."""
    rng = np.random.default_rng(SEED)
    cols = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets",
            "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s",
            "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Mean", "Bwd IAT Mean",
            "Min Packet Length", "Max Packet Length", "Packet Length Mean",
            "Packet Length Std", "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
            "Average Packet Size", "Subflow Fwd Bytes", "Init_Win_bytes_forward",
            "act_data_pkt_fwd", "Active Mean", "Idle Mean"]
    fams = {"BENIGN": 40000, "DoS": 6000, "DDoS": 6000, "PortScan": 5000,
            "Brute Force": 4000, "Web Attack": 1500, "Bot": 1500, "Infiltration": 800}
    def block(src, shift):
        parts = []
        for fam, n in fams.items():
            base = rng.lognormal(mean=3 + shift, sigma=1.2, size=(n, len(cols)))
            if fam != "BENIGN":
                base *= (1.4 + 0.15 * hash(fam) % 5)
            d = pd.DataFrame(base, columns=cols)
            d["Label"] = fam if fam != "BENIGN" else "BENIGN"
            parts.append(d)
        out = pd.concat(parts, ignore_index=True)
        return out
    p17 = os.path.join(DIR_2017, "SYNTHETIC_2017.csv")
    p18 = os.path.join(DIR_2018, "SYNTHETIC_2018.csv")
    block("2017", 0.0).to_csv(p17, index=False)
    block("2018", 0.4).to_csv(p18, index=False)
    return [p17], [p18]


PATHS_2017, PATHS_2018 = download_datasets()
mem_report("after download")


# ============================================================
# 4. DATASET DISCOVERY  (recursive CSV / parquet finder)
# ============================================================
def discover_tables(root):
    files = []
    for ext in ("*.csv", "*.parquet"):
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(set(files))

print("\nDiscovered CIC-IDS2017 files:")
for p in discover_tables(DIR_2017):
    print("  ", p)
print("Discovered CSE-CIC-IDS2018 files:")
for p in discover_tables(DIR_2018):
    print("  ", p)


# ============================================================
# 5. COLUMN NORMALIZATION  +  2017 <-> 2018 SYNONYM MAP
# ============================================================
import re

def norm_col(c):
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")

# maps CIC-2018 short CICFlowMeter names -> CIC-2017 canonical (already norm_col'd)
ALIAS = {
    "dst_port": "destination_port",
    "tot_fwd_pkts": "total_fwd_packets",
    "tot_bwd_pkts": "total_backward_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets",
    "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "fwd_pkt_len_max": "fwd_packet_length_max",
    "fwd_pkt_len_min": "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std": "fwd_packet_length_std",
    "bwd_pkt_len_max": "bwd_packet_length_max",
    "bwd_pkt_len_min": "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std": "bwd_packet_length_std",
    "flow_byts_s": "flow_bytes_s",
    "flow_pkts_s": "flow_packets_s",
    "fwd_iat_tot": "fwd_iat_total",
    "bwd_iat_tot": "bwd_iat_total",
    "fwd_header_len": "fwd_header_length",
    "bwd_header_len": "bwd_header_length",
    "fwd_pkts_s": "fwd_packets_s",
    "bwd_pkts_s": "bwd_packets_s",
    "pkt_len_min": "min_packet_length",
    "pkt_len_max": "max_packet_length",
    "pkt_len_mean": "packet_length_mean",
    "pkt_len_std": "packet_length_std",
    "pkt_len_var": "packet_length_variance",
    "fin_flag_cnt": "fin_flag_count",
    "syn_flag_cnt": "syn_flag_count",
    "rst_flag_cnt": "rst_flag_count",
    "psh_flag_cnt": "psh_flag_count",
    "ack_flag_cnt": "ack_flag_count",
    "urg_flag_cnt": "urg_flag_count",
    "cwe_flag_count": "cwe_flag_count",
    "ece_flag_cnt": "ece_flag_count",
    "pkt_size_avg": "average_packet_size",
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "fwd_byts_b_avg": "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg": "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg": "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg": "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "down_up_ratio": "down_up_ratio",
}
# identifier / leakage / non-generalizing columns to drop on sight
DROP_COLS = {"flow_id", "src_ip", "source_ip", "dst_ip", "destination_ip", "src_port",
             "source_port", "timestamp", "fwd_header_length_1", "protocol",
             "unnamed_0", "id"}

def canon(c):
    n = norm_col(c)
    return ALIAS.get(n, n)


# ============================================================
# 6. LABEL HARMONIZATION  (raw label -> attack family)
# ============================================================
def normalize_label(raw):
    s = str(raw).strip().lower()
    s = s.replace("\x96", "-").replace("–", "-").replace("_", " ")
    if s in ("", "nan", "label"):
        return None
    if "benign" in s or s == "normal":
        return "BENIGN"
    if "heartbleed" in s:
        return "Heartbleed"
    if "bot" in s:
        return "Bot"
    if "portscan" in s or "port scan" in s:
        return "PortScan"
    if "xss" in s or "sql" in s or "web attack" in s or "brute force -web" in s:
        return "Web Attack"
    if "ddos" in s or "ddos attack" in s:
        return "DDoS"
    if "dos" in s or "goldeneye" in s or "hulk" in s or "slowloris" in s or "slowhttptest" in s:
        return "DoS"
    if "patator" in s or "bruteforce" in s or "brute-force" in s or "brute force" in s:
        return "Brute Force"
    if "infil" in s:
        return "Infiltration"
    return "Other"


# ============================================================
# 7. MITRE ATT&CK MAPPING LAYER
# ============================================================
# NOTE: these are ATT&CK Enterprise *interpretations* of dataset labels, not ground truth
# telemetry. Where a confident mapping is not possible we say so instead of inventing one.
MITRE_MAPPING = {
    "Brute Force":  {"tactic": "Credential Access", "technique": "Brute Force",
                     "technique_id": "T1110", "confidence": "High"},
    "DoS":          {"tactic": "Impact", "technique": "Endpoint Denial of Service",
                     "technique_id": "T1499", "confidence": "High"},
    "DDoS":         {"tactic": "Impact", "technique": "Network Denial of Service",
                     "technique_id": "T1498", "confidence": "High"},
    "PortScan":     {"tactic": "Discovery", "technique": "Network Service Discovery",
                     "technique_id": "T1046", "confidence": "High"},
    "Bot":          {"tactic": "Command and Control", "technique": "Application Layer Protocol",
                     "technique_id": "T1071", "confidence": "Medium"},
    "Web Attack":   {"tactic": "Initial Access", "technique": "Exploit Public-Facing Application",
                     "technique_id": "T1190", "confidence": "Medium"},
    "Heartbleed":   {"tactic": "Initial Access", "technique": "Exploit Public-Facing Application",
                     "technique_id": "T1190", "confidence": "High"},
    "Infiltration": {"tactic": "Lateral Movement", "technique": "Exploitation of Remote Services",
                     "technique_id": "T1210", "confidence": "Low"},
    "Other":        {"tactic": "Not confidently mapped", "technique": "Not confidently mapped",
                     "technique_id": "Not confidently mapped", "confidence": "None"},
    "BENIGN":       {"tactic": "N/A (benign traffic)", "technique": "N/A",
                     "technique_id": "N/A", "confidence": "N/A"},
}
# finer-grained hints for the detailed table (keyed by substrings of the raw label)
MITRE_GRANULAR = {
    "ftp-patator": "T1110.001 (Password Guessing)", "ssh-patator": "T1110.001 (Password Guessing)",
    "ftp-bruteforce": "T1110.001 (Password Guessing)", "ssh-bruteforce": "T1110.001 (Password Guessing)",
    "sql injection": "T1190 (Exploit Public-Facing Application)",
    "sql": "T1190 (Exploit Public-Facing Application)",
    "xss": "T1059.007 (JavaScript) / T1189 (Drive-by) - low confidence",
    "hoic": "T1498.001 (Direct Network Flood)", "loic": "T1498.001 (Direct Network Flood)",
}

def mitre_lookup(norm_attack):
    return MITRE_MAPPING.get(norm_attack, MITRE_MAPPING["Other"])


# ============================================================
# 8 + 9.  MEMORY-EFFICIENT LOADING  +  CLEANING  +  STRATIFIED SAMPLING
#         +  FEATURE ALIGNMENT -> COMMON_FEATURES  (with parquet cache)
# ============================================================
# Speed notes:
#   * once a class has hit its cap, later rows of that class are dropped BEFORE the
#     expensive numeric coercion -> chunks from already-covered files skip in ~1 s.
#   * normalize_label() is applied only to the handful of *unique* label strings.
#   * per-chunk drop_duplicates is removed (done once at the end on the small frame).
#   * the harmonized combined frame is cached to parquet -> re-runs skip loading entirely.
NONFEAT = {"original_label", "attack_category", "binary_label", "dataset_source"}
CACHE_PARQUET = os.path.join(DATA_ROOT,
                             f"combined_{'sample' if USE_SAMPLING else 'full'}_{MAX_ROWS_PER_DATASET}.parquet")

def _read_iter(path):
    """Yield dataframe chunks from a csv or parquet file."""
    if path.endswith(".parquet"):
        yield pd.read_parquet(path)
    else:
        for enc in ("latin-1", "utf-8"):
            try:
                for ch in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False,
                                      encoding=enc, on_bad_lines="skip"):
                    yield ch
                return
            except Exception:
                continue


def _cap_for(lab):
    if not USE_SAMPLING:
        return 10 ** 15
    return BENIGN_CAP if lab == "BENIGN" else MAX_ROWS_PER_CLASS


def load_dataset(paths, source_name):
    """Stream every file, normalize columns/labels, clean, keep a stratified cap per class."""
    print(f"\n=== Loading {source_name}  ({len(paths)} file(s)) ===")
    kept, counts, total_seen = {}, {}, 0
    t_start = time.time()
    for fi, path in enumerate(paths, 1):
        for chunk in _read_iter(path):
            total_seen += len(chunk)
            chunk.columns = [canon(c) for c in chunk.columns]
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]
            lab_col = next((c for c in ("label", "attack", "class") if c in chunk.columns), None)
            if lab_col is None:
                continue
            # ---- labels (map only the unique strings) ----
            raw = chunk[lab_col].astype("string").str.strip()
            uniq = {u: normalize_label(u) for u in raw.dropna().unique()}
            lab = raw.map(uniq)
            keep = lab.notna().to_numpy()
            if not keep.any():
                continue
            chunk = chunk.loc[keep].copy()
            chunk["__label__"] = lab.to_numpy()[keep]
            chunk["__orig__"] = raw.to_numpy()[keep]
            # ---- drop rows of already-full classes BEFORE numeric work ----
            if USE_SAMPLING:
                full = {l for l, c in counts.items() if c >= _cap_for(l)}
                if full:
                    chunk = chunk[~chunk["__label__"].isin(full)]
            if len(chunk) == 0:
                continue
            drop = [c for c in DROP_COLS if c in chunk.columns]
            if lab_col not in ("__label__", "__orig__"):
                drop.append(lab_col)
            chunk = chunk.drop(columns=drop, errors="ignore")
            feat = [c for c in chunk.columns if c not in ("__label__", "__orig__")]
            # ---- numeric coercion + inf/nan handling ----
            X = chunk[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            good = X.notna().all(axis=1).to_numpy()
            if not good.any():
                continue
            X = X.loc[good].astype(np.float32)
            X["__label__"] = chunk["__label__"].to_numpy()[good]
            X["__orig__"] = chunk["__orig__"].to_numpy()[good]
            # ---- stratified capping ----
            for lb, grp in X.groupby("__label__", sort=False):
                cap = _cap_for(lb); have = counts.get(lb, 0)
                if have >= cap:
                    continue
                take = grp.iloc[: cap - have] if len(grp) > cap - have else grp
                kept.setdefault(lb, []).append(take)
                counts[lb] = have + len(take)
            del chunk, X; gc.collect()
        print(f"   file {fi}/{len(paths)}  scanned {total_seen:,}  kept {sum(counts.values()):,}  "
              f"({time.time()-t_start:.0f}s)")
    df = pd.concat([d for lst in kept.values() for d in lst], ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    df["dataset_source"] = source_name
    print(f"   -> kept {len(df):,} rows  |  class counts {df['__label__'].value_counts().to_dict()}")
    gc.collect()
    return df


def build_combined_dataset():
    """Harmonize both datasets into one frame. Returns (data, COMMON_FEATURES)."""
    df17 = load_dataset(PATHS_2017, "CIC-IDS2017")
    df18 = load_dataset(PATHS_2018, "CSE-CIC-IDS2018")
    meta = {"__label__", "__orig__", "dataset_source"}
    n17 = {c for c in df17.columns if c not in meta and pd.api.types.is_numeric_dtype(df17[c])}
    n18 = {c for c in df18.columns if c not in meta and pd.api.types.is_numeric_dtype(df18[c])}
    common = sorted(n17 & n18)
    probe = pd.concat([df17[common].head(20000), df18[common].head(20000)])
    common = [c for c in common if probe[c].nunique() > 2]
    print(f"\nCIC-IDS2017 numeric features {len(n17)} | CSE-CIC-IDS2018 {len(n18)} | "
          f"COMMON_FEATURES retained {len(common)}")
    print(", ".join(common))
    d = pd.concat([df17[common + list(meta)], df18[common + list(meta)]], ignore_index=True)
    del df17, df18, probe; gc.collect()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=common).drop_duplicates()
    d["original_label"] = d["__orig__"].astype(str).str.replace(r"[^\x00-\x7f]+", " - ", regex=True)
    d["attack_category"] = d["__label__"]
    d["binary_label"] = np.where(d["attack_category"] == "BENIGN", "BENIGN", "ATTACK")
    d = d.drop(columns=["__label__", "__orig__"]).reset_index(drop=True)
    return d, common


# ---- load (from parquet cache if present, else build + cache) ----
t0 = time.time()
if os.path.exists(CACHE_PARQUET):
    print(f"\nUsing cached combined dataset: {CACHE_PARQUET}  "
          "(delete this file to force a fresh rebuild)")
    data = pd.read_parquet(CACHE_PARQUET)
    COMMON_FEATURES = [c for c in data.columns if c not in NONFEAT]
else:
    data, COMMON_FEATURES = build_combined_dataset()
    try:
        data.to_parquet(CACHE_PARQUET, index=False)
        print(f"Cached combined dataset -> {CACHE_PARQUET}")
    except Exception as e:
        print("   (parquet cache skipped:", e, ")")
LOAD_TIME = time.time() - t0

print(f"\nCombined dataset shape        : {data.shape}   (load {LOAD_TIME:.0f}s)")
print(f"Binary distribution           : {data['binary_label'].value_counts().to_dict()}")
print(f"Attack-category distribution  :\n{data['attack_category'].value_counts()}")
print(f"By dataset source             :\n{data.groupby('dataset_source')['binary_label'].value_counts()}")
mem_report("after combine")


# ============================================================
# 10. MITRE ATT&CK MAPPING TABLE
# ============================================================
rows = []
for orig, grp in data.groupby("original_label"):
    norm = normalize_label(orig) or "Other"
    m = mitre_lookup(norm)
    gran = next((v for k, v in MITRE_GRANULAR.items() if k in str(orig).lower()), "")
    rows.append({"Dataset Attack": orig, "Normalized Attack": norm,
                 "MITRE Tactic": m["tactic"], "MITRE Technique": m["technique"],
                 "Technique ID": m["technique_id"], "Confidence": m["confidence"],
                 "Sub-technique hint": gran, "Flow Count": len(grp)})
mitre_df = pd.DataFrame(rows).sort_values("Flow Count", ascending=False).reset_index(drop=True)
mitre_df.to_csv(os.path.join(OUTPUT_DIR, "mitre_attack_mapping.csv"), index=False)
print("\nMITRE ATT&CK mapping (dataset label -> ATT&CK interpretation):")
print(mitre_df.to_string(index=False))


# ============================================================
# 11. TRAIN / VALIDATION / TEST SPLIT  (stratified, leakage-reduced)
# ============================================================
data = data.reset_index(drop=True)
y_bin = (data["binary_label"] == "ATTACK").astype(int).values
strat = data["binary_label"] + "|" + data["dataset_source"]

idx = np.arange(len(data))
tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=strat)
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED,
                                  stratify=strat.iloc[tr_idx])

X_all = data[COMMON_FEATURES].astype(np.float32)
print(f"\nTrain: {len(tr_idx):,}   Val: {len(va_idx):,}   Test: {len(te_idx):,}")
print(f"Train attack rate {y_bin[tr_idx].mean():.3f} | Val {y_bin[va_idx].mean():.3f} | Test {y_bin[te_idx].mean():.3f}")


# ============================================================
# 12. PREPROCESSING PIPELINE  (fit on TRAIN ONLY -> no leakage)
# ============================================================
preprocessor = ColumnTransformer(
    [("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]), COMMON_FEATURES)],
    remainder="drop")

t0 = time.time()
Xtr = preprocessor.fit_transform(X_all.iloc[tr_idx]).astype(np.float32)
Xva = preprocessor.transform(X_all.iloc[va_idx]).astype(np.float32)
Xte = preprocessor.transform(X_all.iloc[te_idx]).astype(np.float32)
PREP_TIME = time.time() - t0

ytr, yva, yte = y_bin[tr_idx], y_bin[va_idx], y_bin[te_idx]
src_te = data["dataset_source"].values[te_idx]
cat_te = data["attack_category"].values[te_idx]
orig_te = data["original_label"].values[te_idx]
joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.pkl"))


# ============================================================
# 13. FEATURE ANALYSIS  (correlation, mutual information, RF importance, PCA)
# ============================================================
print("\nComputing feature analysis ...")
_s = min(20000, len(Xtr))
_sub = np.random.RandomState(SEED).choice(len(Xtr), _s, replace=False)
mi = mutual_info_classif(Xtr[_sub], ytr[_sub], random_state=SEED)
mi_s = pd.Series(mi, index=COMMON_FEATURES).sort_values(ascending=False)

rf_fs = RandomForestClassifier(n_estimators=120, max_depth=18, n_jobs=-1, random_state=SEED)
rf_fs.fit(Xtr[_sub], ytr[_sub])
imp_s = pd.Series(rf_fs.feature_importances_, index=COMMON_FEATURES).sort_values(ascending=False)

corr = pd.DataFrame(Xtr[_sub], columns=COMMON_FEATURES).corr()
pca = PCA(n_components=2, random_state=SEED).fit(Xtr[_sub])
pca_xy = pca.transform(Xtr[_sub])

print("Top-15 features by mutual information:\n", mi_s.head(15).round(4).to_string())
print("Top-15 features by Random Forest importance:\n", imp_s.head(15).round(4).to_string())
TOP_FEATURES = list(imp_s.head(10).index)


# ============================================================
# 14. CNN  1D  DEEP FEATURE EXTRACTOR
# ============================================================
from tensorflow.keras import layers, models, callbacks

def build_cnn(n_features, embed_dim=CNN_EMBED_DIM):
    inp = layers.Input(shape=(n_features, 1), name="flow_input")
    x = layers.Conv1D(64, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv1D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    emb = layers.Dense(embed_dim, activation="relu", name="embedding")(x)
    d = layers.Dropout(0.3)(emb)
    out = layers.Dense(1, activation="sigmoid", name="detection")(d)
    model = models.Model(inp, out, name="CNN_feature_extractor")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

cnn = build_cnn(Xtr.shape[1])
cnn.summary()

cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
t0 = time.time()
hist = cnn.fit(Xtr[..., None], ytr,
               validation_data=(Xva[..., None], yva),
               epochs=CNN_EPOCHS, batch_size=CNN_BATCH, class_weight=cw, verbose=2,
               callbacks=[callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                                                  monitor="val_loss")])
CNN_TIME = time.time() - t0
cnn.save(os.path.join(OUTPUT_DIR, "cnn_feature_extractor.keras"))

embedder = models.Model(cnn.input, cnn.get_layer("embedding").output, name="cnn_embedder")
Etr = embedder.predict(Xtr[..., None], batch_size=1024, verbose=0)
Eva = embedder.predict(Xva[..., None], batch_size=1024, verbose=0)
Ete = embedder.predict(Xte[..., None], batch_size=1024, verbose=0)
print(f"\nCNN embeddings: train {Etr.shape}  val {Eva.shape}  test {Ete.shape}")

# CNN-only baseline (reference row - NOT part of the 3-way hybrid comparison)
cnn_score_te = cnn.predict(Xte[..., None], batch_size=1024, verbose=0).ravel()
cnn_pred_te = (cnn_score_te >= 0.5).astype(int)


# ============================================================
# 15. METRIC HELPERS
# ============================================================
def compute_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = np.nan
    try:
        prauc = average_precision_score(y_true, y_score)
    except Exception:
        prauc = np.nan
    return dict(Accuracy=accuracy_score(y_true, y_pred),
               Precision=precision_score(y_true, y_pred, zero_division=0),
               Recall=recall_score(y_true, y_pred, zero_division=0),
               F1=f1_score(y_true, y_pred, zero_division=0),
               Specificity=spec, FPR=fpr, FNR=fnr,
               ROC_AUC=roc, PR_AUC=prauc,
               MCC=matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
               Balanced_Acc=balanced_accuracy_score(y_true, y_pred),
               TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


# ============================================================
# 16. TRAIN THE THREE HYBRIDS  (CNN embedding -> classical model)
# ============================================================
def sub_for_tuning(E, y):
    if len(E) <= TUNE_SUBSAMPLE:
        return E, y
    s = np.random.RandomState(SEED).choice(len(E), TUNE_SUBSAMPLE, replace=False)
    return E[s], y[s]

Es, ys = sub_for_tuning(Etr, ytr)
timings, models_store, scores_store, preds_store = {}, {}, {}, {}


def proba_attack(clf, E):
    """P(attack) for any estimator - works whether or not predict_proba exists."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(E)[:, 1]
    if hasattr(clf, "decision_function"):
        d = np.ravel(clf.decision_function(E))
        return 1.0 / (1.0 + np.exp(-d))          # logistic squash of the SVM margin
    return clf.predict(E).astype(float)


# ---- CNN + SVM --------------------------------------------------------------
# A genuine Support Vector Machine. RBF is O(n^2) in samples, so on large training
# sets we fit it on a stratified subsample (standard practice for SVM-based IDS) and
# say so explicitly. Robust across every scikit-learn version (no CalibratedClassifierCV
# signature issues).
print("\n--- Training CNN + SVM ---")
t0 = time.time()
if len(Etr) <= SVM_RBF_MAX_TRAIN:
    E_fit, y_fit = Etr, ytr
    SVM_KIND = f"RBF SVC, full train ({len(E_fit):,} samples)"
else:
    n_sub = min(30_000, len(Etr))
    E_fit, _, y_fit, _ = train_test_split(Etr, ytr, train_size=n_sub, stratify=ytr,
                                          random_state=SEED)
    SVM_KIND = f"RBF SVC, stratified subsample ({n_sub:,} of {len(Etr):,} train samples)"
try:
    svm = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
              class_weight="balanced", random_state=SEED)
    svm.fit(E_fit, y_fit)
except Exception as e:
    print("   RBF SVC unavailable -> LinearSVC fallback:", repr(e))
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=5000)
    svm.fit(Etr, ytr)
    SVM_KIND = "LinearSVC (margin -> logistic score)"
timings["CNN+SVM"] = time.time() - t0
models_store["CNN+SVM"] = svm
scores_store["CNN+SVM"] = proba_attack(svm, Ete)
preds_store["CNN+SVM"] = (scores_store["CNN+SVM"] >= 0.5).astype(int)
print("   ", SVM_KIND)

# ---- CNN + Decision Tree ---------------------------------------------------
print("--- Training CNN + Decision Tree ---")
t0 = time.time()
dt_grid = GridSearchCV(
    DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    {"max_depth": [12, 20, 30, None], "min_samples_split": [2, 10, 50],
     "min_samples_leaf": [1, 5, 20], "criterion": ["gini", "entropy"]},
    scoring="f1", cv=3, n_jobs=-1)
dt_grid.fit(Es, ys)
dt = DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **dt_grid.best_params_)
dt.fit(Etr, ytr)
timings["CNN+DecisionTree"] = time.time() - t0
models_store["CNN+DecisionTree"] = dt
scores_store["CNN+DecisionTree"] = dt.predict_proba(Ete)[:, 1]
preds_store["CNN+DecisionTree"] = dt.predict(Ete)
print("   best params:", dt_grid.best_params_)

# ---- CNN + Random Forest -------------------------------------------------
print("--- Training CNN + Random Forest ---")
t0 = time.time()
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    {"n_estimators": [150, 250, 400], "max_depth": [16, 24, 32, None],
     "min_samples_split": [2, 10, 25], "min_samples_leaf": [1, 4, 10],
     "max_features": ["sqrt", "log2", 0.5]},
    n_iter=8, scoring="f1", cv=3, n_jobs=-1, random_state=SEED)
rf_search.fit(Es, ys)
rf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
                            **rf_search.best_params_)
rf.fit(Etr, ytr)
timings["CNN+RandomForest"] = time.time() - t0
models_store["CNN+RandomForest"] = rf
scores_store["CNN+RandomForest"] = rf.predict_proba(Ete)[:, 1]
preds_store["CNN+RandomForest"] = rf.predict(Ete)
print("   best params:", rf_search.best_params_)


# ============================================================
# 17. EVALUATION  (overall + per-dataset + per-attack + inference time)
# ============================================================
INF_TIME = {}
for name, mdl in models_store.items():
    t0 = time.time(); _ = mdl.predict(Ete[:5000]); INF_TIME[name] = (time.time() - t0) / min(5000, len(Ete))

results = {name: compute_metrics(yte, preds_store[name], scores_store[name])
           for name in models_store}
results["CNN (baseline)"] = compute_metrics(yte, cnn_pred_te, cnn_score_te)

results_df = pd.DataFrame(results).T
results_df["Train_Time_s"] = pd.Series({**timings, "CNN (baseline)": CNN_TIME})
results_df["Infer_us_per_flow"] = pd.Series({k: v * 1e6 for k, v in INF_TIME.items()})
order = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR", "FNR",
         "ROC_AUC", "PR_AUC", "MCC", "Balanced_Acc",
         "TP", "TN", "FP", "FN", "Train_Time_s", "Infer_us_per_flow"]
results_df = results_df[order]
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
print("\n================  MODEL COMPARISON (test set)  ================")
print(results_df.round(4).to_string())

# ---- per-dataset generalization ----------------------------------------
per_ds_rows = []
for name in models_store:
    for ds in np.unique(src_te):
        m = src_te == ds
        r = compute_metrics(yte[m], preds_store[name][m], scores_store[name][m])
        per_ds_rows.append({"Model": name, "Dataset": ds, **{k: r[k] for k in
                            ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "MCC"]}})
per_ds_df = pd.DataFrame(per_ds_rows)
per_ds_df.to_csv(os.path.join(OUTPUT_DIR, "per_dataset_performance.csv"), index=False)
print("\n----  Per-dataset generalization  ----")
print(per_ds_df.round(4).to_string(index=False))

# ---- per-attack-category detection performance -----------------------
per_atk_rows = []
for cat in sorted(set(cat_te)):
    if cat == "BENIGN":
        continue
    m = cat_te == cat
    if m.sum() < 10:
        continue
    for name in models_store:
        rec = recall_score((cat_te[m] != "BENIGN").astype(int), preds_store[name][m], zero_division=0)
        per_atk_rows.append({"Attack": cat, "Model": name, "Test_Flows": int(m.sum()),
                             "Detection_Rate(Recall)": rec})
per_atk_df = pd.DataFrame(per_atk_rows)
per_atk_df.to_csv(os.path.join(OUTPUT_DIR, "per_attack_performance.csv"), index=False)
print("\n----  Per-attack detection rate  ----")
if len(per_atk_df):
    print(per_atk_df.pivot(index="Attack", columns="Model",
                           values="Detection_Rate(Recall)").round(3).to_string())

# ---- classification reports -------------------------------------------
rep_rows = []
for name in models_store:
    rep = classification_report(yte, preds_store[name], target_names=["BENIGN", "ATTACK"],
                                output_dict=True, zero_division=0)
    for cls in ["BENIGN", "ATTACK"]:
        rep_rows.append({"Model": name, "Class": cls, **rep[cls]})
pd.DataFrame(rep_rows).to_csv(os.path.join(OUTPUT_DIR, "classification_reports.csv"), index=False)


# ============================================================
# 18. STATIC VISUALIZATIONS  (saved to ./IDS_Results/)
# ============================================================
sns.set_style("whitegrid")
HYB = list(models_store.keys())
PALETTE = {"CNN+SVM": "#00d4ff", "CNN+DecisionTree": "#ffb000", "CNN+RandomForest": "#ff4b6e"}

def savefig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=130, bbox_inches="tight")
    plt.close(fig)

# --- dataset analysis ---
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
data["dataset_source"].value_counts().plot.bar(ax=ax[0, 0], color=["#00d4ff", "#ff4b6e"]); ax[0, 0].set_title("Rows per dataset source")
data["binary_label"].value_counts().plot.bar(ax=ax[0, 1], color=["#2ecc71", "#e74c3c"]); ax[0, 1].set_title("ATTACK vs BENIGN")
data["attack_category"].value_counts().plot.bar(ax=ax[0, 2]); ax[0, 2].set_title("Attack categories"); ax[0, 2].tick_params(axis="x", rotation=45)
pd.crosstab(data["attack_category"], data["dataset_source"]).plot.bar(ax=ax[1, 0]); ax[1, 0].set_title("Attack category by dataset"); ax[1, 0].tick_params(axis="x", rotation=45)
(data["attack_category"].value_counts(normalize=True) * 100).plot.barh(ax=ax[1, 1]); ax[1, 1].set_title("Class imbalance (%)")
data.groupby("dataset_source")["binary_label"].value_counts().unstack().plot.bar(ax=ax[1, 2]); ax[1, 2].set_title("Attack/benign split by dataset")
fig.suptitle("Dataset Intelligence", fontsize=15); savefig(fig, "01_dataset_analysis.png")

# --- feature analysis ---
fig, ax = plt.subplots(2, 2, figsize=(18, 12))
mi_s.head(20).iloc[::-1].plot.barh(ax=ax[0, 0], color="#00d4ff"); ax[0, 0].set_title("Top-20 features - Mutual Information")
imp_s.head(20).iloc[::-1].plot.barh(ax=ax[0, 1], color="#ff4b6e"); ax[0, 1].set_title("Top-20 features - Random Forest importance")
top_corr = imp_s.head(15).index
sns.heatmap(corr.loc[top_corr, top_corr], cmap="mako", ax=ax[1, 0], cbar=True); ax[1, 0].set_title("Correlation (top-15 features)")
sc = ax[1, 1].scatter(pca_xy[:, 0], pca_xy[:, 1], c=ytr[_sub], cmap="coolwarm", s=4, alpha=0.4)
ax[1, 1].set_title(f"PCA 2D (var explained {pca.explained_variance_ratio_.sum():.2f})"); plt.colorbar(sc, ax=ax[1, 1])
fig.suptitle("Feature Analysis", fontsize=15); savefig(fig, "02_feature_analysis.png")

# --- CNN learning curves ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
ax[0].set_title("CNN accuracy"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
ax[1].set_title("CNN loss"); ax[1].legend()
savefig(fig, "03_cnn_learning_curves.png")

# --- confusion matrices (counts + row-normalized) ---
fig, ax = plt.subplots(2, 3, figsize=(18, 11))
for j, name in enumerate(HYB):
    cm = confusion_matrix(yte, preds_store[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0, j],
                xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"])
    ax[0, j].set_title(f"{name} - counts")
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2%", cmap="Purples", ax=ax[1, j],
                xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"])
    ax[1, j].set_title(f"{name} - row-normalized")
fig.suptitle("Confusion Matrices", fontsize=15); savefig(fig, "04_confusion_matrices.png")

# --- ROC + PR ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for name in HYB:
    fpr_c, tpr_c, _ = roc_curve(yte, scores_store[name])
    ax[0].plot(fpr_c, tpr_c, label=f"{name} (AUC={results[name]['ROC_AUC']:.3f})", color=PALETTE[name])
    pr, rc, _ = precision_recall_curve(yte, scores_store[name])
    ax[1].plot(rc, pr, label=f"{name} (AP={results[name]['PR_AUC']:.3f})", color=PALETTE[name])
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()
ax[1].set_title("Precision-Recall"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()
savefig(fig, "05_roc_pr_curves.png")

# --- metric comparison bars ---
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR",
                   "ROC_AUC", "PR_AUC", "MCC"]
fig, ax = plt.subplots(3, 3, figsize=(17, 13))
for a, mname in zip(ax.ravel(), metrics_to_plot):
    vals = [results[n][mname] for n in HYB]
    a.bar(HYB, vals, color=[PALETTE[n] for n in HYB]); a.set_title(mname)
    a.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Model Comparison by Metric", fontsize=15); savefig(fig, "06_metric_comparison.png")

# --- radar chart ---
radar_metrics = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC_AUC", "PR_AUC", "MCC"]
ang = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(8, 8)); axp = plt.subplot(polar=True)
for name in HYB:
    vals = [max(0, results[name][m]) for m in radar_metrics]; vals += vals[:1]
    axp.plot(ang, vals, label=name, color=PALETTE[name]); axp.fill(ang, vals, alpha=0.1, color=PALETTE[name])
axp.set_xticks(ang[:-1]); axp.set_xticklabels(radar_metrics); axp.set_title("Model Performance Radar (visualization only)")
axp.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1)); savefig(fig, "07_radar.png")

# --- performance heatmap ---
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(results_df.loc[HYB, ["Accuracy", "Precision", "Recall", "F1", "Specificity",
                                 "ROC_AUC", "PR_AUC", "MCC"]].astype(float),
            annot=True, fmt=".3f", cmap="viridis", ax=ax)
ax.set_title("Performance heatmap"); savefig(fig, "08_performance_heatmap.png")

# --- error analysis (TP/TN/FP/FN) ---
err = results_df.loc[HYB, ["TP", "TN", "FP", "FN"]].astype(int)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
err[["FP", "FN"]].plot.bar(ax=ax[0], color=["#ffb000", "#ff4b6e"]); ax[0].set_title("False Positives vs False Negatives")
err.plot.bar(ax=ax[1]); ax[1].set_title("Confusion counts per model")
savefig(fig, "09_error_analysis.png")

# --- per dataset + per attack ---
fig, ax = plt.subplots(1, 2, figsize=(17, 6))
per_ds_df.pivot(index="Dataset", columns="Model", values="F1").plot.bar(ax=ax[0]); ax[0].set_title("F1 by dataset")
if len(per_atk_df):
    per_atk_df.pivot(index="Attack", columns="Model", values="Detection_Rate(Recall)").plot.bar(ax=ax[1])
    ax[1].set_title("Detection rate by attack category"); ax[1].tick_params(axis="x", rotation=45)
savefig(fig, "10_dataset_and_attack_performance.png")

# --- MITRE visuals ---
mm = mitre_df[mitre_df["Normalized Attack"] != "BENIGN"]
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
mm.groupby("MITRE Tactic")["Flow Count"].sum().sort_values().plot.barh(ax=ax[0], color="#00d4ff"); ax[0].set_title("Flows per MITRE tactic")
mm.groupby("Technique ID")["Flow Count"].sum().sort_values().plot.barh(ax=ax[1], color="#ff4b6e"); ax[1].set_title("Flows per MITRE technique")
piv = pd.crosstab(mm["Normalized Attack"], mm["MITRE Tactic"], values=mm["Flow Count"], aggfunc="sum").fillna(0)
sns.heatmap(piv, annot=True, fmt=".0f", cmap="rocket", ax=ax[2]); ax[2].set_title("Attack category x MITRE tactic")
fig.suptitle("MITRE ATT&CK Interpretation of Dataset Labels", fontsize=15); savefig(fig, "11_mitre_attack.png")

print(f"\nSaved {len(glob.glob(os.path.join(OUTPUT_DIR,'*.png')))} figures to ./{OUTPUT_DIR}/")


# ============================================================
# 18B. ACCURACY STUDY  (accuracy-focused comparison + charts)
#   NOTE: accuracy is reported next to F1 / recall / ROC-AUC on purpose - on
#   imbalanced IDS traffic, accuracy alone is optimistic. The "representation"
#   experiment is a drop-in slot: replace the CNN embedding with a quantum /
#   QFT-enhanced feature map and re-run for a like-for-like accuracy comparison.
# ============================================================
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

acc_df = pd.DataFrame(columns=["Train", "Validation", "Test", "CV_mean", "CV_std", "Improvement_vs_CNN_%"])
rep_df = pd.DataFrame(columns=["Representation", "Classifier", "Test Accuracy"])
comp   = {}

try:
    _n_acc = min(18_000, len(Etr))
    _sa = np.random.RandomState(SEED).choice(len(Etr), _n_acc, replace=False)
    _cv_est = {
        "CNN+SVM":          SVC(kernel="rbf", C=2.0, gamma="scale",
                                class_weight="balanced", random_state=SEED),
        "CNN+DecisionTree": clone(models_store["CNN+DecisionTree"]),
        "CNN+RandomForest": clone(models_store["CNN+RandomForest"]),
    }
    _rows = []
    for name in HYB:
        mdl = models_store[name]
        tr = accuracy_score(ytr[_sa], mdl.predict(Etr[_sa]))
        va = accuracy_score(yva, mdl.predict(Eva))
        te = float(results[name]["Accuracy"])
        try:
            cv = cross_val_score(_cv_est[name], Etr[_sa], ytr[_sa], cv=3,
                                 scoring="accuracy", n_jobs=-1)
            cvm, cvs = float(cv.mean()), float(cv.std())
        except Exception:
            cvm, cvs = np.nan, np.nan
        _rows.append({"Model": name, "Train": tr, "Validation": va, "Test": te,
                      "CV_mean": cvm, "CV_std": cvs})
    _cnn_tr = (cnn.predict(Xtr[_sa][..., None], verbose=0).ravel() >= 0.5).astype(int)
    _cnn_va = (cnn.predict(Xva[..., None], verbose=0).ravel() >= 0.5).astype(int)
    _rows.append({"Model": "CNN (baseline)",
                  "Train": accuracy_score(ytr[_sa], _cnn_tr),
                  "Validation": accuracy_score(yva, _cnn_va),
                  "Test": float(results["CNN (baseline)"]["Accuracy"]),
                  "CV_mean": np.nan, "CV_std": np.nan})
    acc_df = pd.DataFrame(_rows).set_index("Model")
    _base = float(acc_df.loc["CNN (baseline)", "Test"])
    acc_df["Improvement_vs_CNN_%"] = (acc_df["Test"] - _base) / _base * 100.0
    acc_df.to_csv(os.path.join(OUTPUT_DIR, "accuracy_study.csv"))

    # ---- model-complexity proxy ----
    def _complexity(mdl):
        if isinstance(mdl, RandomForestClassifier):
            return int(sum(e.tree_.node_count for e in mdl.estimators_))
        if isinstance(mdl, DecisionTreeClassifier):
            return int(mdl.tree_.node_count)
        if hasattr(mdl, "support_"):
            return int(len(mdl.support_))
        if hasattr(mdl, "coef_"):
            return int(np.size(mdl.coef_))
        return 1
    comp = {name: _complexity(models_store[name]) for name in HYB}
    comp["CNN (baseline)"] = int(cnn.count_params())

    # ---- representation experiment: raw features vs CNN embedding ----
    _nr = min(30_000, len(Xtr))
    _sr = np.random.RandomState(SEED).choice(len(Xtr), _nr, replace=False)
    _stst = np.random.RandomState(SEED + 1).choice(len(Xte), min(20_000, len(Xte)), replace=False)
    _rep_factory = {
        "SVM":           lambda: SVC(kernel="rbf", C=2.0, gamma="scale",
                                     class_weight="balanced", random_state=SEED),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=20, class_weight="balanced",
                                                        random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=SEED),
    }
    _rr = []
    for rep_name, Xr_tr, Xr_te in [("Raw scaled features", Xtr, Xte),
                                   ("CNN embedding", Etr, Ete)]:
        for clf_name, fac in _rep_factory.items():
            clf = fac(); clf.fit(Xr_tr[_sr], ytr[_sr])
            a = accuracy_score(yte[_stst], clf.predict(Xr_te[_stst]))
            _rr.append({"Representation": rep_name, "Classifier": clf_name, "Test Accuracy": a})
            del clf; gc.collect()
    rep_df = pd.DataFrame(_rr)
    rep_df.to_csv(os.path.join(OUTPUT_DIR, "representation_accuracy.csv"), index=False)

    print("\n----  ACCURACY STUDY  ----")
    print(acc_df.round(4).to_string())
    print("\nRepresentation experiment (fixed 30k-train / 20k-test subsample):")
    print(rep_df.pivot(index="Classifier", columns="Representation", values="Test Accuracy").round(4).to_string())

    # ---- charts 12-18 ----
    _ORD = HYB + ["CNN (baseline)"]
    _CL  = [PALETTE.get(m, "#8b9dc3") for m in _ORD]

    fig, ax = plt.subplots(figsize=(9, 5))
    _b = ax.bar(_ORD, acc_df.loc[_ORD, "Test"], color=_CL)
    ax.set_ylim(max(0.0, float(acc_df.loc[_ORD, "Test"].min()) - 0.04), 1.002)
    ax.set_ylabel("Test accuracy"); ax.set_title("Model Accuracy Comparison  (held-out test set)")
    for r in _b:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15); savefig(fig, "12_accuracy_by_model.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    _xx = np.arange(len(_ORD)); _w = 0.27
    for i, col in enumerate(["Train", "Validation", "Test"]):
        ax.bar(_xx + (i - 1) * _w, acc_df.loc[_ORD, col], _w, label=col)
    ax.set_xticks(_xx); ax.set_xticklabels(_ORD, rotation=15); ax.set_ylabel("Accuracy")
    ax.set_ylim(max(0.0, float(acc_df.loc[_ORD, ["Train", "Validation", "Test"]].min().min()) - 0.04), 1.002)
    ax.set_title("Training vs Validation vs Testing Accuracy"); ax.legend()
    savefig(fig, "13_train_val_test_accuracy.png")

    _cvd = acc_df.loc[HYB]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(_cvd.index, _cvd["CV_mean"], yerr=_cvd["CV_std"], capsize=6,
           color=[PALETTE[m] for m in _cvd.index])
    ax.set_ylabel("3-fold CV accuracy"); ax.set_title("Cross-Validation Accuracy  (mean +/- std)")
    ax.set_ylim(max(0.0, float((_cvd["CV_mean"] - _cvd["CV_std"]).min()) - 0.03), 1.002)
    for i, (_m, r) in enumerate(_cvd.iterrows()):
        ax.text(i, r["CV_mean"], f"{r['CV_mean']:.4f}", ha="center", va="bottom", fontsize=9)
    savefig(fig, "14_cv_accuracy.png")

    _imp = acc_df.loc[HYB, "Improvement_vs_CNN_%"]
    fig, ax = plt.subplots(figsize=(8, 5))
    _bars = ax.bar(_imp.index, _imp.values, color=["#2ecc71" if v >= 0 else "#e74c3c" for v in _imp.values])
    ax.axhline(0, color="k", lw=1); ax.set_ylabel("Accuracy improvement over CNN baseline (%)")
    ax.set_title("Accuracy Improvement (%)  vs CNN softmax baseline")
    for r in _bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():+.2f}%",
                ha="center", va="bottom" if r.get_height() >= 0 else "top", fontsize=9)
    savefig(fig, "15_accuracy_improvement.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in _ORD:
        ax.scatter(comp.get(m, 1), acc_df.loc[m, "Test"], s=150,
                   color=PALETTE.get(m, "#8b9dc3"), zorder=3, edgecolor="white")
        ax.annotate(m, (comp.get(m, 1), acc_df.loc[m, "Test"]), xytext=(7, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel("Model complexity  (tree nodes / support vectors / CNN params, log scale)")
    ax.set_ylabel("Test accuracy"); ax.set_title("Accuracy vs Model Complexity")
    savefig(fig, "16_accuracy_vs_complexity.png")

    _rpv = rep_df.pivot(index="Classifier", columns="Representation", values="Test Accuracy")
    _rpv = _rpv[[c for c in ["Raw scaled features", "CNN embedding"] if c in _rpv.columns]]
    fig, ax = plt.subplots(figsize=(9, 5))
    _rpv.plot.bar(ax=ax, color=["#8892b0", "#22d3ee"][: _rpv.shape[1]])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(max(0.0, float(_rpv.min().min()) - 0.04), 1.002)
    ax.set_title("Accuracy: Classical (raw features) vs CNN-Enhanced (embedding)\n"
                 "[drop-in slot for a quantum / QFT-enhanced representation]")
    ax.tick_params(axis="x", rotation=0); ax.legend(title="Representation")
    for _cont in ax.containers:
        ax.bar_label(_cont, fmt="%.3f", fontsize=8)
    savefig(fig, "17_representation_accuracy.png")

    _srt = acc_df.loc[HYB, "Test"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    _bh = ax.barh(_srt.index, _srt.values, color=[PALETTE[m] for m in _srt.index])
    ax.set_xlabel("Test accuracy"); ax.set_xlim(max(0.0, float(_srt.min()) - 0.03), 1.002)
    ax.set_title("Accuracy Comparison Across Classifiers  (CNN-enhanced hybrids)")
    for r in _bh:
        ax.text(r.get_width(), r.get_y() + r.get_height() / 2, f" {r.get_width():.4f}",
                va="center", fontsize=9)
    savefig(fig, "18_accuracy_across_classifiers.png")

    print(f"Accuracy-study charts 12-18 saved to ./{OUTPUT_DIR}/")
except Exception as _e:
    print("Accuracy study skipped:", repr(_e)); traceback.print_exc()


# ============================================================
# 19. MODEL LEADERBOARD  (F1 -> Recall -> ROC-AUC -> low FPR)
# ============================================================
lb = results_df.loc[HYB].copy()
lb["rank_key"] = list(zip(lb["F1"], lb["Recall"], lb["ROC_AUC"], -lb["FPR"]))
lb = lb.sort_values("rank_key", ascending=False)
BEST_MODEL = lb.index[0]
medals = ["🥇", "🥈", "🥉"]
print("\n==================  MODEL LEADERBOARD  ==================")
for i, name in enumerate(lb.index):
    r = results[name]
    print(f" {medals[i]}  {name:20s}  F1={r['F1']:.4f}  Recall={r['Recall']:.4f}  "
          f"ROC-AUC={r['ROC_AUC']:.4f}  FPR={r['FPR']:.4f}")

best_est = models_store[BEST_MODEL]
joblib.dump({"model": best_est, "name": BEST_MODEL, "common_features": COMMON_FEATURES,
             "top_features": TOP_FEATURES, "embed_dim": CNN_EMBED_DIM},
            os.path.join(OUTPUT_DIR, "best_model.pkl"))
joblib.dump({n: m for n, m in models_store.items()}, os.path.join(OUTPUT_DIR, "all_hybrid_models.pkl"))


# ============================================================
# 20. AUTOMATIC RESEARCH REPORT
# ============================================================
lowest_fpr = min(HYB, key=lambda n: results[n]["FPR"])
best_recall = max(HYB, key=lambda n: results[n]["Recall"])
fastest = min(HYB, key=lambda n: timings[n])
top_techniques = mm.groupby(["Technique ID", "MITRE Technique"])["Flow Count"].sum().sort_values(ascending=False)

_acc_txt = acc_df.round(4).to_string() if len(acc_df) else "(accuracy study unavailable)"
if len(rep_df):
    _rp = rep_df.pivot(index="Classifier", columns="Representation", values="Test Accuracy")
    if {"CNN embedding", "Raw scaled features"} <= set(_rp.columns):
        _rp["gain"] = _rp["CNN embedding"] - _rp["Raw scaled features"]
    _rep_txt = _rp.round(4).to_string()
else:
    _rep_txt = "(representation experiment unavailable)"

report = f"""
============================================================
EXPERIMENT SUMMARY   {'(*** SYNTHETIC FALLBACK DATA ***)' if DATASET_IS_SYNTHETIC else ''}
============================================================
Datasets            : CIC-IDS2017 + CSE-CIC-IDS2018 (harmonized, {len(COMMON_FEATURES)} common features)
Sampling            : USE_SAMPLING={USE_SAMPLING}
Total samples used  : {len(data):,}   (train {len(tr_idx):,} / val {len(va_idx):,} / test {len(te_idx):,})
Features (CNN input): {len(COMMON_FEATURES)}
Attack categories   : {sorted(set(data['attack_category']) - {'BENIGN'})}
Class balance (bin) : {data['binary_label'].value_counts().to_dict()}
Experiment seed     : {SEED}
GPU                 : {bool(GPUS)}

------------------  MEASURED RESULTS  ------------------
{results_df.loc[HYB].round(4).to_string()}

Best model (F1>Recall>ROC-AUC>low FPR) : {BEST_MODEL}
Best F1                                : {results[BEST_MODEL]['F1']:.4f}
Best ROC-AUC                           : {max(results[n]['ROC_AUC'] for n in HYB):.4f}
Lowest FPR                             : {lowest_fpr}  ({results[lowest_fpr]['FPR']:.4f})
Best attack recall                     : {best_recall}  ({results[best_recall]['Recall']:.4f})
Fastest to train                       : {fastest}  ({timings[fastest]:.1f}s)
Preprocessing time                     : {PREP_TIME:.1f}s   CNN training: {CNN_TIME:.1f}s

------------------  ACCURACY STUDY (measured)  ------------------
{_acc_txt}

Representation experiment (test accuracy; 'gain' = CNN embedding - raw features):
{_rep_txt}

Most represented MITRE techniques (by flow count, dataset-label interpretation):
{top_techniques.head(6).to_string()}

------------------  RESEARCH INTERPRETATION  ------------------
1. Best performer: {BEST_MODEL}. On CNN embeddings the classical heads mainly differ in how they
   carve the learned feature space; the winner above is the one that maximised F1 while keeping
   recall and ROC-AUC high on the held-out test set.
2. Likely reason: the 1D-CNN already linearises much of the class structure, so a
   {'high-capacity ensemble (Random Forest) exploits residual non-linear interactions' if 'Random' in BEST_MODEL else 'margin-based / axis-aligned head generalises well on the compact embedding'}.
3. Lowest false-positive rate: {lowest_fpr} - most attractive for a SOC that is alert-fatigued.
4. Best attack recall: {best_recall} - most attractive when missing an intrusion is costlier than a false alarm.
5. Cross-dataset behaviour: see per_dataset_performance.csv / figure 10. Gaps between the
   CIC-IDS2017 and CSE-CIC-IDS2018 columns quantify generalisation; larger gaps = more
   dataset-specific (feature-distribution) overfitting.
6. MITRE ATT&CK: the traffic is dominated by {top_techniques.index[0][0]} ({top_techniques.index[0][1]})
   and other Impact / Discovery / Credential-Access techniques - typical of the flood, scan and
   brute-force scenarios these datasets emphasise.
7. Limitations: flow-level CICFlowMeter features only (no payload); binary head; class caps under
   USE_SAMPLING; MITRE mapping is an interpretation of labels, not verified adversary telemetry;
   Infiltration mapping is low-confidence; the two datasets share tooling so cross-dataset transfer
   is optimistic vs. truly unseen networks.
8. Future work: multiclass attack-family head, temporal/sequence models (LSTM/Transformer on flow
   windows), adversarial-robustness testing, true leave-one-dataset-out evaluation, SHAP explanations
   per alert, and streaming/online deployment.
============================================================
"""
print(report)
with open(os.path.join(OUTPUT_DIR, "experiment_report.txt"), "w", encoding="utf-8") as fh:
    fh.write(report)

print("Artifacts written to ./%s/ :" % OUTPUT_DIR)
for f in sorted(os.listdir(OUTPUT_DIR)):
    print("   ", f)


# ============================================================
# 21. INTERACTIVE SOC DASHBOARD  (Gradio + Plotly - dark "threat-intel centre" theme)
# ============================================================
CLR   = ["#22d3ee", "#f59e0b", "#fb7185", "#34d399", "#a78bfa", "#60a5fa"]
INK   = "#0b0f17"
GRID  = "rgba(120,150,180,0.14)"
FONT  = "Inter, 'Segoe UI', system-ui, sans-serif"

def style_fig(f, h=430, title=None):
    f.update_layout(
        template="plotly_dark", height=h, colorway=CLR,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#c7d2df", size=12),
        title=dict(text=title, font=dict(size=15, color="#e5edf5")) if title else None,
        margin=dict(l=48, r=24, t=52 if title else 24, b=44),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor="#111827", font_size=12, font_family=FONT))
    f.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    f.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return f

def _vc(series, name):
    v = series.value_counts()
    return pd.DataFrame({name: v.index.astype(str), "count": v.values})


# ---- accuracy-study figures (data computed in section 18B) ----
def _acc_reset():
    return acc_df.reset_index().rename(columns={"index": "Model"}) if len(acc_df) else \
        pd.DataFrame(columns=["Model", "Train", "Validation", "Test", "CV_mean", "CV_std",
                              "Improvement_vs_CNN_%"])

def fig_acc_bar():
    d = _acc_reset()[["Model", "Test"]]
    return style_fig(px.bar(d, x="Model", y="Test", color="Model", text_auto=".4f",
                            color_discrete_map=PALETTE), 420, "Model Accuracy Comparison (test set)")

def fig_acc_tvt():
    d = _acc_reset().melt(id_vars="Model", value_vars=["Train", "Validation", "Test"],
                          var_name="Split", value_name="Accuracy")
    return style_fig(px.bar(d, x="Model", y="Accuracy", color="Split", barmode="group",
                            text_auto=".3f"), 420, "Training vs Validation vs Testing Accuracy")

def fig_acc_cv():
    d = _acc_reset()
    d = d[d["Model"].isin(HYB)]
    return style_fig(px.bar(d, x="Model", y="CV_mean", color="Model", error_y="CV_std",
                            text_auto=".4f", color_discrete_map=PALETTE),
                     420, "Cross-Validation Accuracy (3-fold, mean +/- std)")

def fig_acc_impr():
    d = _acc_reset()
    d = d[d["Model"].isin(HYB)]
    return style_fig(px.bar(d, x="Model", y="Improvement_vs_CNN_%", text_auto="+.2f",
                            color="Improvement_vs_CNN_%", color_continuous_scale="RdYlGn"),
                     420, "Accuracy Improvement (%) over CNN baseline")

def fig_acc_cplx():
    d = pd.DataFrame({"Model": list(comp.keys()),
                      "Complexity": [comp[m] for m in comp],
                      "Test accuracy": [float(acc_df.loc[m, "Test"]) if m in acc_df.index else np.nan
                                        for m in comp]})
    d["sz"] = 22
    f = px.scatter(d, x="Complexity", y="Test accuracy", text="Model", color="Model", size="sz",
                   log_x=True, color_discrete_map=PALETTE)
    f.update_traces(textposition="top center")
    return style_fig(f, 440, "Accuracy vs Model Complexity (log scale)")

def fig_rep():
    return style_fig(px.bar(rep_df, x="Classifier", y="Test Accuracy", color="Representation",
                            barmode="group", text_auto=".3f",
                            color_discrete_map={"Raw scaled features": "#8892b0",
                                                "CNN embedding": "#22d3ee"}),
                     440, "Classical (raw features) vs CNN-Enhanced (embedding) Accuracy")


BEST_F1  = results[BEST_MODEL]["F1"]
BEST_ROC = max(results[n]["ROC_AUC"] for n in HYB)
BEST_REC = max(results[n]["Recall"] for n in HYB)
LOW_FPR  = min(results[n]["FPR"] for n in HYB)
TOTAL    = len(data); ATK = int((data["binary_label"] == "ATTACK").sum()); BEN = TOTAL - ATK
ATK_RATE = ATK / TOTAL * 100

def _card(label, value, sub="", accent="#22d3ee"):
    return (f"<div class='kpi'><div class='kpi-bar' style='background:{accent}'></div>"
            f"<div class='kpi-l'>{label}</div><div class='kpi-v'>{value}</div>"
            f"<div class='kpi-s'>{sub}</div></div>")

def kpi_html():
    sev = "#fb7185" if ATK_RATE >= 40 else "#f59e0b" if ATK_RATE >= 15 else "#34d399"
    cards = [
        _card("TOTAL FLOWS", f"{TOTAL:,}", "harmonized 2017 + 2018", "#60a5fa"),
        _card("BENIGN", f"{BEN:,}", f"{100-ATK_RATE:.1f}% of traffic", "#34d399"),
        _card("ATTACK", f"{ATK:,}", f"{ATK_RATE:.1f}% of traffic", sev),
        _card("ATTACK SURFACE", f"{len(set(data['attack_category'])-{'BENIGN'})} families",
              f"{len(COMMON_FEATURES)} shared features", "#a78bfa"),
        _card("BEST MODEL", BEST_MODEL.replace("CNN+", "CNN + "), "F1-ranked winner", "#22d3ee"),
        _card("BEST F1", f"{BEST_F1:.3f}", f"ROC-AUC {BEST_ROC:.3f}", "#22d3ee"),
        _card("BEST RECALL", f"{BEST_REC:.3f}", "attack detection rate", "#f59e0b"),
        _card("LOWEST FPR", f"{LOW_FPR:.3f}", "false-alarm floor", "#34d399"),
    ]
    return "<div class='kpi-wrap'>" + "".join(cards) + "</div>"

def leaderboard_html():
    med = ["🥇", "🥈", "🥉"]
    order = sorted(HYB, key=lambda n: (results[n]["F1"], results[n]["Recall"],
                                       results[n]["ROC_AUC"], -results[n]["FPR"]), reverse=True)
    rows = ""
    for i, n in enumerate(order):
        r = results[n]
        rows += (f"<div class='lb-row'><span class='lb-rank'>{med[i]}</span>"
                 f"<span class='lb-name'>{n}</span>"
                 f"<span class='lb-m'>F1 <b>{r['F1']:.3f}</b></span>"
                 f"<span class='lb-m'>Recall <b>{r['Recall']:.3f}</b></span>"
                 f"<span class='lb-m'>ROC-AUC <b>{r['ROC_AUC']:.3f}</b></span>"
                 f"<span class='lb-m'>FPR <b>{r['FPR']:.3f}</b></span></div>")
    return f"<div class='lb'>{rows}</div>"

def fig_gauge():
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=BEST_F1 * 100, number={"suffix": " F1", "font": {"size": 26}},
        title={"text": f"Detection posture · {BEST_MODEL}", "font": {"size": 13}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#c7d2df"},
               "bar": {"color": "#22d3ee"},
               "steps": [{"range": [0, 60], "color": "#3f1d2b"},
                         {"range": [60, 85], "color": "#3d3218"},
                         {"range": [85, 100], "color": "#123a2a"}],
               "threshold": {"line": {"color": "#fb7185", "width": 3}, "value": 90}}))
    return style_fig(f, 300)

def fig_attack_donut():
    d = _vc(data["attack_category"], "category")
    f = px.pie(d, names="category", values="count", hole=0.62)
    f.update_traces(textposition="outside", textinfo="label+percent")
    return style_fig(f, 430, "Attack-category mix")

def fig_bin_by_ds():
    d = data.groupby(["dataset_source", "binary_label"]).size().reset_index(name="flows")
    f = px.bar(d, x="dataset_source", y="flows", color="binary_label", barmode="group",
               color_discrete_map={"BENIGN": "#34d399", "ATTACK": "#fb7185"})
    return style_fig(f, 430, "Benign vs attack by dataset")

def fig_model_cmp(metric):
    d = results_df.loc[HYB, metric].reset_index(); d.columns = ["Model", metric]
    f = px.bar(d, x="Model", y=metric, color="Model", text_auto=".3f",
               color_discrete_map=PALETTE)
    f.update_traces(textposition="outside")
    return style_fig(f, 430, f"Model comparison · {metric}")

def fig_roc():
    f = go.Figure()
    for name in HYB:
        fc, tc, _ = roc_curve(yte, scores_store[name])
        f.add_scatter(x=fc, y=tc, mode="lines", name=f"{name}  (AUC {results[name]['ROC_AUC']:.3f})",
                      line=dict(color=PALETTE[name], width=2.5))
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                  line=dict(dash="dash", color="#64748b"))
    return style_fig(f, 430, "ROC — all hybrids")

def fig_pr():
    f = go.Figure()
    for name in HYB:
        pr, rc, _ = precision_recall_curve(yte, scores_store[name])
        f.add_scatter(x=rc, y=pr, mode="lines", name=f"{name}  (AP {results[name]['PR_AUC']:.3f})",
                      line=dict(color=PALETTE[name], width=2.5))
    return style_fig(f, 430, "Precision–Recall — all hybrids")

def fig_radar():
    f = go.Figure()
    for name in HYB:
        vals = [max(0, results[name][m]) for m in radar_metrics]
        f.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_metrics + [radar_metrics[0]],
                                    fill="toself", name=name, line=dict(color=PALETTE[name])))
    f.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(range=[0, 1], gridcolor=GRID),
                               angularaxis=dict(gridcolor=GRID)))
    return style_fig(f, 430, "Capability radar (visual only)")

def fig_confusion(model):
    cm = confusion_matrix(yte, preds_store[model])
    cmn = cm / cm.sum(1, keepdims=True)
    f = px.imshow(cmn, x=["BENIGN", "ATTACK"], y=["BENIGN", "ATTACK"], color_continuous_scale="Tealrose",
                  text_auto=".2%", aspect="auto")
    return style_fig(f, 430, f"Confusion (row-normalized) · {model}")

def fig_ds_metric(metric):
    f = px.bar(per_ds_df, x="Dataset", y=metric, color="Model", barmode="group",
               text_auto=".3f", color_discrete_map=PALETTE)
    return style_fig(f, 430, f"Dataset-wise generalization · {metric}")

def fig_mitre_tactic():
    d = mm.groupby("MITRE Tactic")["Flow Count"].sum().sort_values().reset_index()
    return style_fig(px.bar(d, x="Flow Count", y="MITRE Tactic", orientation="h",
                            color="Flow Count", color_continuous_scale="Cividis"),
                     430, "Flows per MITRE tactic")

def fig_mitre_tech():
    d = mm.groupby(["Technique ID", "MITRE Technique"])["Flow Count"].sum().sort_values().reset_index()
    d["lbl"] = d["Technique ID"] + " · " + d["MITRE Technique"]
    return style_fig(px.bar(d, x="Flow Count", y="lbl", orientation="h",
                            color="Flow Count", color_continuous_scale="Plasma"),
                     430, "Flows per MITRE technique")

def fig_mitre_heat():
    piv2 = pd.crosstab(mm["Normalized Attack"], mm["MITRE Tactic"],
                       values=mm["Flow Count"], aggfunc="sum").fillna(0)
    return style_fig(px.imshow(piv2, text_auto=".0f", aspect="auto",
                               color_continuous_scale="Turbo"), 460, "Attack family × MITRE tactic")

def fig_per_attack():
    if not len(per_atk_df):
        return style_fig(go.Figure(), 430, "Per-attack detection (insufficient test samples)")
    f = px.bar(per_atk_df, x="Attack", y="Detection_Rate(Recall)", color="Model", barmode="group",
               text_auto=".2f", color_discrete_map=PALETTE)
    return style_fig(f, 460, "Detection rate by attack category")

def ds_stats():
    g = data.groupby("dataset_source").agg(flows=("binary_label", "size"),
                                           attacks=("binary_label", lambda s: (s == "ATTACK").sum()))
    g["benign"] = g["flows"] - g["attacks"]
    g["attack_rate_%"] = (g["attacks"] / g["flows"] * 100).round(2)
    return g.reset_index()

def threat_detail(cat):
    m = cat_te == cat
    info = mitre_lookup(cat)
    n = int((data["attack_category"] == cat).sum())
    perf = {f"{name} recall": round(float(recall_score((cat_te[m] != "BENIGN").astype(int),
             preds_store[name][m], zero_division=0)), 3) for name in HYB} if m.sum() else {}
    return pd.DataFrame([{"Attack": cat, "Flows (total)": n, "% of dataset": round(n / TOTAL * 100, 2),
                          "Flows (test)": int(m.sum()),
                          "MITRE Tactic": info["tactic"], "MITRE Technique": info["technique"],
                          "Technique ID": info["technique_id"], "Confidence": info["confidence"],
                          **perf}])

# ---- manual prediction ----
raw_med = X_all.iloc[tr_idx].median()

def predict_flow(model_choice, *vals):
    row = raw_med.copy()
    for f, v in zip(TOP_FEATURES, vals):
        if v is not None:
            row[f] = v
    x = preprocessor.transform(pd.DataFrame([row])[COMMON_FEATURES].astype(np.float32))
    emb = embedder.predict(x[..., None], verbose=0)
    proba = float(proba_attack(models_store[model_choice], emb)[0])
    is_atk = proba >= 0.5
    label = "ATTACK" if is_atk else "BENIGN"
    if   proba >= .90: risk, rc = "CRITICAL", "#fb7185"
    elif proba >= .70: risk, rc = "HIGH", "#f59e0b"
    elif proba >= .50: risk, rc = "ELEVATED", "#fbbf24"
    elif proba >= .30: risk, rc = "GUARDED", "#38bdf8"
    else:              risk, rc = "LOW", "#34d399"
    bar = int(round(proba * 100))
    mitre = ""
    if is_atk:
        items = ""
        for (tid, tname) in list(top_techniques.head(4).index):
            sub = mm[mm["Technique ID"] == tid]
            tac = sub["MITRE Tactic"].iloc[0] if len(sub) else ""
            items += f"<li><b>{tid}</b> · {tname} <span class='muted'>({tac})</span></li>"
        mitre = (f"<div class='mitre-box'><div class='mitre-h'>MITRE ATT&CK context "
                 f"(most represented in these datasets)</div><ul>{items}</ul></div>")
    verdict_c = "#fb7185" if is_atk else "#34d399"
    return (f"<div class='verdict'><div class='v-badge' style='background:{verdict_c}1a;"
            f"border-color:{verdict_c}'><span style='color:{verdict_c}'>{label}</span></div>"
            f"<div class='v-body'>"
            f"<div class='v-row'><span>Attack probability</span><b>{proba:.3f}</b></div>"
            f"<div class='meter'><div class='meter-fill' style='width:{bar}%;background:{rc}'></div></div>"
            f"<div class='v-row'><span>Risk level</span>"
            f"<b><span class='chip' style='background:{rc}22;color:{rc};border-color:{rc}'>{risk}</span></b></div>"
            f"<div class='v-row'><span>Classifier head</span><b>{model_choice}</b></div>"
            f"<div class='v-row'><span>Pipeline</span><b>preprocess → CNN → 64-D embedding → head</b></div>"
            f"</div></div>{mitre}")

try:
    import gradio as gr
    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Orbitron:wght@600;800&display=swap');
    .gradio-container{background:#070b12 !important;color:#c7d2df !important;font-family:Inter,system-ui,sans-serif !important;max-width:1400px !important}
    footer{display:none !important}
    #hero{background:linear-gradient(120deg,#0b1220 0%,#0e2233 45%,#111827 100%);
      border:1px solid #1e3a4c;border-radius:16px;padding:22px 26px;margin-bottom:6px;
      box-shadow:0 0 40px rgba(34,211,238,.08) inset}
    #hero h1{font-family:Orbitron,sans-serif !important;font-size:24px !important;letter-spacing:2px;
      color:#e6f6ff !important;margin:0 0 4px 0;text-shadow:0 0 18px rgba(34,211,238,.45)}
    #hero .sub{color:#7dd3fc;font-size:13px;letter-spacing:1px}
    #hero .tag{display:inline-block;margin-top:10px;padding:4px 10px;border-radius:999px;font-size:11px;
      background:#0c2a1f;color:#34d399;border:1px solid #135e43}
    .kpi-wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:14px 0}
    .kpi{position:relative;background:linear-gradient(160deg,#0c1420,#111c2b);border:1px solid #1c2c3e;
      border-radius:14px;padding:14px 14px 12px 18px;overflow:hidden}
    .kpi-bar{position:absolute;left:0;top:0;bottom:0;width:4px}
    .kpi-l{font-size:10.5px;letter-spacing:1.4px;color:#7c93a8;text-transform:uppercase}
    .kpi-v{font-size:22px;font-weight:800;color:#eef5fb;margin-top:6px;font-family:Orbitron,sans-serif}
    .kpi-s{font-size:11px;color:#63788c;margin-top:3px}
    .lb{display:flex;flex-direction:column;gap:8px;margin-top:6px}
    .lb-row{display:flex;align-items:center;gap:14px;background:#0c1420;border:1px solid #1c2c3e;
      border-radius:12px;padding:10px 16px;flex-wrap:wrap}
    .lb-rank{font-size:20px}.lb-name{font-weight:800;color:#e6f6ff;min-width:160px}
    .lb-m{font-size:12px;color:#8fa6ba}.lb-m b{color:#22d3ee}
    .verdict{display:flex;gap:18px;align-items:stretch;margin-top:8px;flex-wrap:wrap}
    .v-badge{display:flex;align-items:center;justify-content:center;min-width:150px;border:2px solid;
      border-radius:14px;font-family:Orbitron,sans-serif;font-size:22px;font-weight:800;letter-spacing:2px}
    .v-body{flex:1;min-width:280px;background:#0c1420;border:1px solid #1c2c3e;border-radius:14px;padding:14px 18px}
    .v-row{display:flex;justify-content:space-between;padding:6px 0;font-size:13px;color:#93a7ba}
    .v-row b{color:#e6f6ff}
    .meter{height:10px;background:#132030;border-radius:999px;overflow:hidden;margin:6px 0 10px}
    .meter-fill{height:100%;border-radius:999px;transition:width .4s}
    .chip{padding:2px 10px;border-radius:999px;border:1px solid;font-size:11px;font-weight:700}
    .mitre-box{margin-top:14px;background:#0c1420;border:1px solid #24384c;border-radius:14px;padding:12px 18px}
    .mitre-h{color:#7dd3fc;font-size:12px;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase}
    .mitre-box ul{margin:0;padding-left:18px}.mitre-box li{padding:3px 0;font-size:13px;color:#cbd8e4}
    .muted{color:#6b7f92}
    .tabs button{font-weight:700 !important;letter-spacing:.5px}
    .tab-nav button.selected{color:#22d3ee !important;border-bottom:2px solid #22d3ee !important}
    label span{color:#8fa6ba !important}
    """
    tag = "SYNTHETIC FALLBACK DATA" if DATASET_IS_SYNTHETIC else "LIVE · CIC-IDS2017 + CSE-CIC-IDS2018"
    with gr.Blocks(css=CSS, title="Cyber Threat Intelligence & Intrusion Detection Center",
                   theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate")) as demo:
        gr.HTML(f"""<div id='hero'>
            <h1>CYBER THREAT INTELLIGENCE &amp; INTRUSION DETECTION CENTER</h1>
            <div class='sub'>AI-POWERED NETWORK INTRUSION DETECTION &amp; MITRE ATT&amp;CK ANALYTICS</div>
            <div class='sub' style='color:#5f7488'>Hybrid CNN Feature Learning + Classical Machine Learning</div>
            <span class='tag'>● {tag}</span></div>""")

        with gr.Tab("🛰  Executive Dashboard"):
            gr.HTML(kpi_html())
            with gr.Row():
                gr.Plot(fig_gauge()); gr.Plot(fig_attack_donut()); gr.Plot(fig_bin_by_ds())
            gr.HTML("<h3 style='color:#7dd3fc;font-family:Orbitron'>MODEL LEADERBOARD</h3>")
            gr.HTML(leaderboard_html())
            with gr.Row():
                gr.Plot(fig_model_cmp("F1")); gr.Plot(fig_roc())

        with gr.Tab("🗄  Dataset Intelligence"):
            gr.Dataframe(ds_stats(), label="Per-dataset statistics", interactive=False)
            gr.Dataframe(_vc(data["attack_category"], "attack_family"),
                         label="Class distribution (combined)", interactive=False)
            gr.Plot(style_fig(px.histogram(
                data.melt(id_vars="dataset_source", value_vars=TOP_FEATURES[:4]),
                x="value", color="dataset_source", facet_col="variable", facet_col_wrap=2,
                log_y=True, color_discrete_sequence=["#22d3ee", "#fb7185"]),
                520, "Top feature distributions — 2017 vs 2018"))

        with gr.Tab("📊  Model Comparison"):
            met = gr.Dropdown(["Accuracy", "Precision", "Recall", "F1", "Specificity", "FPR",
                               "ROC_AUC", "PR_AUC", "MCC", "Balanced_Acc"], value="F1", label="Metric")
            pc = gr.Plot(fig_model_cmp("F1"))
            met.change(fig_model_cmp, met, pc)
            with gr.Row():
                gr.Plot(fig_pr()); gr.Plot(fig_radar())
            cm_sel = gr.Dropdown(HYB, value=BEST_MODEL, label="Confusion matrix — model")
            cm_plot = gr.Plot(fig_confusion(BEST_MODEL))
            cm_sel.change(fig_confusion, cm_sel, cm_plot)
            gr.Dataframe(results_df.loc[HYB].round(4).reset_index().rename(columns={"index": "Model"}),
                         label="Full measured metric table", interactive=False)

        with gr.Tab("📈  Accuracy Study"):
            gr.HTML("<p style='color:#8fa6ba'>Accuracy is shown next to F1 / recall / ROC-AUC on "
                    "purpose - on imbalanced IDS traffic, accuracy alone is optimistic. The "
                    "<b>representation panel</b> is a drop-in slot: swap the CNN embedding for a "
                    "quantum / QFT-enhanced feature map and re-run for a like-for-like comparison.</p>")
            with gr.Row():
                gr.Plot(fig_acc_bar()); gr.Plot(fig_acc_tvt())
            with gr.Row():
                gr.Plot(fig_acc_cv()); gr.Plot(fig_acc_impr())
            with gr.Row():
                gr.Plot(fig_acc_cplx()); gr.Plot(fig_rep())
            gr.Dataframe(_acc_reset().round(4),
                         label="Accuracy study - train / val / test / CV / improvement",
                         interactive=False)
            gr.Dataframe(rep_df.round(4),
                         label="Representation experiment - raw features vs CNN embedding",
                         interactive=False)

        with gr.Tab("🎯  MITRE ATT&CK"):
            with gr.Row():
                gr.Plot(fig_mitre_tactic()); gr.Plot(fig_mitre_tech())
            gr.Plot(fig_mitre_heat())
            f_ta = gr.Dropdown(["(all)"] + sorted(mm["MITRE Tactic"].unique()), value="(all)",
                               label="Filter by MITRE tactic")
            mt = gr.Dataframe(mitre_df, label="Dataset label → MITRE ATT&CK interpretation",
                              interactive=False)
            f_ta.change(lambda ta: mitre_df if ta == "(all)" else mitre_df[mitre_df["MITRE Tactic"] == ta],
                        f_ta, mt)

        with gr.Tab("🔬  Threat Analysis"):
            cats = sorted(set(data["attack_category"]) - {"BENIGN"})
            cc = gr.Dropdown(cats, value=cats[0], label="Attack category")
            td = gr.Dataframe(threat_detail(cats[0]), interactive=False)
            cc.change(threat_detail, cc, td)
            dm = gr.Dropdown(["F1", "Recall", "Precision", "FPR", "ROC_AUC", "MCC"], value="Recall",
                             label="Dataset-wise metric")
            dsp = gr.Plot(fig_ds_metric("Recall"))
            dm.change(fig_ds_metric, dm, dsp)
            gr.Plot(fig_per_attack())

        with gr.Tab("⚡  Live / Manual Prediction"):
            gr.HTML("<p style='color:#8fa6ba'>Enter values for the most informative flow features "
                    "(the rest default to the training median). The flow is scored through "
                    "<b>preprocess → CNN → 64-D embedding → selected classical head</b>.</p>")
            mc = gr.Radio(HYB, value=BEST_MODEL, label="Classifier head")
            with gr.Row():
                inputs = [gr.Number(value=round(float(raw_med[f]), 4), label=f) for f in TOP_FEATURES]
            out = gr.HTML()
            gr.Button("▶  ANALYZE FLOW", variant="primary").click(predict_flow, [mc] + inputs, out)

        gr.HTML("<p style='text-align:center;color:#4b5f72;font-size:11px;margin-top:16px'>"
                "All figures computed at runtime from the held-out test set · "
                f"seed {SEED} · MITRE mappings are interpretations of dataset labels</p>")

    if LAUNCH_DASHBOARD:
        demo.launch(share=GRADIO_SHARE, debug=False, quiet=True)
except Exception as e:
    print("Dashboard could not launch:", repr(e))
    traceback.print_exc()

print("\n>>> PIPELINE COMPLETE <<<")
