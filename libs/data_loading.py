import pandas as pd
from scipy.io import arff


def load_data(file_path):
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    return df

