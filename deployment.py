import os
import shutil
import pickle
import logging

from config import get_config

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

dataset_csv_path = CONFIG['output_folder_path']
output_model_path = CONFIG['output_model_path']
prod_deployment_path = CONFIG['prod_deployment_path']


def store_model_into_pickle(
    dataset_csv_path,
    output_model_path,
    prod_deployment_path
):  
    """
    Store the model into pickle file and save it in the deployment folder
    """
    ingest_file_path = os.path.join(
        os.getcwd(),
        dataset_csv_path,
        "ingestedfiles.txt"
    )
    model_path = os.path.join(
        os.getcwd(),
        output_model_path,
        'trainedmodel.pkl'
    )
    lastest_score_path = os.path.join(
        os.getcwd(),
        output_model_path,
        'latestscore.txt'
    )
    prod_deployment_path = os.path.join(
        os.getcwd(),
        prod_deployment_path
    )
    
    shutil.copy(ingest_file_path, prod_deployment_path)
    shutil.copy(model_path, prod_deployment_path)
    shutil.copy(lastest_score_path, prod_deployment_path)
    LOGGER.info('All files saved at {}'.format(prod_deployment_path))
    
if __name__ == '__main__':
    store_model_into_pickle(
    dataset_csv_path,
    output_model_path,
    prod_deployment_path
)
