import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob, os, sys
from pathlib import Path
from tqdm import tqdm

import matplotlib._color_data as mcd
import pyarrow.parquet as pq
import click
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.backends.backend_pdf import PdfPages

from SEIR import NPI, setup, file_paths
from SEIR.utils import config
import pathlib
from matplotlib.backends.backend_pdf import PdfPages
from SEIR.setup import _parameter_reduce, npi_load

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import Outcomes.outcomes as outcomes


def get_timeserie(hpar_fn, hnpi_fn, scenario_outcomes,
                                             dates,
                                             places, config):
    npi_config = [config["outcomes"]["interventions"]["settings"][scenario_outcomes], config]
    loaded_values = pq.read_table(hpar_fn).to_pandas()
    hnpi = pq.read_table(hnpi_fn).to_pandas()
    
    npi = NPI.NPIBase.execute(
        npi_config=npi_config[0],
        global_config=npi_config[1],
        geoids=places,
        loaded_df = hnpi
    )
    
    comps = ['incidH', 'incidC', 'incidD']
    results = {}
    
    for new_comp in comps:  

        probabilities = \
        loaded_values[
            (loaded_values['quantity'] == 'probability') &
            (loaded_values['outcome'] == new_comp) &
            (loaded_values['p_comp'] == 0.0) #& (loaded_values['source'] == source)
        ]['value'].to_numpy()

        probabilities[probabilities > 1] = 1
        probabilities[probabilities < 0] = 0
        probabilities = np.repeat(probabilities[:,np.newaxis], len(dates), axis = 1).T  # duplicate in time

        probabilities = _parameter_reduce(probabilities, npi.getReduction(f"{new_comp}::probability".lower()), 1)
        results[new_comp] = probabilities
    return results