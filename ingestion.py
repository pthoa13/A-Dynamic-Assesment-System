import os
import glob
import logging
import numpy as np
import pandas as pd
from config import get_config
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LOGGER = logging.getLogger(__name__)
CONFIG = get_config() 

input_folder_path = CONFIG['input_folder_path']
output_folder_path = CONFIG['output_folder_path']
final_data_path = os.path.join(
    os.getcwd(), 
    output_folder_path, 
    'final_data.csv'
)
ingested_files_path = os.path.join(
    os.getcwd(), 
    output_folder_path,
    'ingestedfiles.txt'
)

def merge_multiple_dataframe(
    input_folder_path,
    ingested_files_path,
    final_data_path
):
    data_path_list = glob.glob(
        os.path.join(
            os.getcwd(),
            input_folder_path,
            "*.csv"
        )
    )
    data_name_list = [name.split("/")[-1] for name in data_path_list]
    
    LOGGER.info(f"Found input data files: {' '.join(data_name_list)})")
    with open(ingested_files_path, "w") as f:
        for name in data_name_list:
            f.write(f"{name}\n")
            
    df = pd.concat(
        [pd.read_csv(path) for path in data_path_list], 
        ignore_index=True
    )
    
    df_processed = df.drop_duplicates()
    df_processed.to_csv(final_data_path, index=False)
    return df_processed
    
    
    
if __name__ == '__main__':
    merge_multiple_dataframe(
        input_folder_path=input_folder_path,
        ingested_files_path=ingested_files_path,
        final_data_path=final_data_path
    )
