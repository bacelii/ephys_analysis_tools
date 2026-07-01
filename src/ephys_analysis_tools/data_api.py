import pickle
from pathlib import Path

def extract_ephys_dict(
    data = None,
    filepath = None,
    verbose = True
    ):
    """
    Purpose
    -------
    To extract a ditionary mapping a neurons name into the ephys time series data
    from the original APL data pickle format

    Original Data format: file > monitor_data > neuron_name , variables > v (the voltage data)
    """
    if data is None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

    ephys_data = dict()

    for monitor_ex in data['monitor_data']:
        neuron_name = monitor_ex['neuron_name']
        voltage_trace = monitor_ex['variables']['v']
        ephys_data[neuron_name] = voltage_trace

    if verbose:
        ex_data = list(ephys_data.values())[0]
        print(f"# of neurons with ephys data = {len(ephys_data)} with size {ex_data.shape}")

    return ephys_data


def example_data_load():

    data_dir = Path("../data/")
    filepath =  data_dir / "run_data_20260417_NFL_Concusion_t_pre.pkl"

    from ephys_analysis_tools import data_api as api
    ephys_data = api.extract_ephys_dict(
        filepath = filepath,
        verbose = True
    )
    
    return ephys_data