import pandas as pd
from scipy.io import arff
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    last_column = df.columns[-1]
    df[last_column] = df[last_column].apply(lambda x: 1 if b'true' in x else 0)


    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.show()


    # Save the figure
    plt.savefig(f'{file_path}.png')

    return df

