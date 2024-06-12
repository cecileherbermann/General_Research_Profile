# Imports
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import sklearn
import warnings
sys.path.insert(0, "../")
import scgpt as scg
# extra dependency for similarity search
import faiss

warnings.filterwarnings("ignore", category=ResourceWarning)

# Change paths according to your scGPT model, adata object and faiss index directory
model_dir = Path("./scGPT_human")
adata = sc.read_h5ad("./data/CFS_all_days_rawcount.h5ad")
gene_col = "index"
index_path="./faiss_index/"

# Subset the AnnData object to include only observations where the 'sample' variable is one day
day0 = adata[adata.obs['sample'] == 'CFS_Day0', :]

# Embedding of adata
day0_embed = scg.tasks.embed_data(
    day0,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
    return_new_adata=True,
)

# Load faiss index
from build_atlas_index_faiss import load_index, vote
index, meta_labels = load_index(
    index_dir=index_path,
    use_config_file=True,
    use_gpu=True,
)

# Add celltype to adata
adata.obs['Day 0'] = voting

# Dimensionality reduction and plotting
sc.pp.neighbors(adata, use_rep="X")
sc.tl.umap(adata)

# Plotting the UMAP and saving the plot
sc.pl.umap(adata, color='Day 0', frameon=False, wspace=0.4, show=False)
plt.savefig("./thesis/umap_day0.png")
