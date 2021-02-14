import os
from datasets.loading import load_data

import os
os.environ['HHC_HOME']= "/Users/forg1ven/Desktop/PC/HypHC"
os.environ['DATAPATH'] = "/Users/forg1ven/Desktop/PC/HypHC/data"
os.environ['SAVEPATH'] = "/Users/forg1ven/Desktop/PC/HypHC/embeddings"

load_data("custom", False)