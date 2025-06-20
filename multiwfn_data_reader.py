import pandas as pd

from my_collections import NAMES_DICT
from my_config import *

class MultiwfnDataReader:
    def __init__(self):
        self.distance_df = self.read_multiwfn_data(MULTIWNF_DISTANCE_FILE_NAME)
        self.integral_norm_df = self.read_multiwfn_data(MULTIWNF_INTEGRAL_NORM_FILE_NAME)
        self.integral_square_df = self.read_multiwfn_data(MULTIWNF_INTEGRAL_SQUARE_FILE_NAME)

        self.general_df = self.merge_dfs()

    def read_multiwfn_data(self, filename):
        with open(filename, 'r') as f:
            value_dict = dict()
            for line in f:
                if line[0] != 'B':
                    continue

                long_name = line.split(': ')[0].split('_')[0]
                value = float(line.split(': ')[1])

                if not long_name in NAMES_DICT:
                    continue

                value_dict[NAMES_DICT[long_name]] = value
            df = pd.DataFrame(list(value_dict.items()), columns=['Key', 'Values'])
            df.sort_values(by=['Key'], inplace=True)
            return df

    def merge_dfs(self):
        tmp = pd.merge(self.distance_df, self.integral_norm_df, how="left", on=["Key"])
        result = pd.merge(tmp, self.integral_square_df, how="left", on=["Key"])
        result.columns = ['Short name', 'Distance', 'Integral norm', 'Integral square']
        return result

if __name__ == "__main__":
    print("__name__ == __main__")
    mdr = MultiwfnDataReader()
else:
    print("multiwfn module has been imported")