import pandas as pd
import preprocessing
import hbac_kmeans
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


DATASET_NAME = "final_preprocessed_compas"

# load data
path = Path(os.getcwd()).absolute()
raw_data = pd.read_csv(str(path) + fr'\Preprocessed_datasets\{DATASET_NAME}.csv', index_col=0)


results = hbac_kmeans.hbac_kmeans(raw_data, show_plot=True)


