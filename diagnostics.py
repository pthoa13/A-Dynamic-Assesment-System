import os
import pickle
import timeit
import logging
import subprocess
import pandas as pd
from ingestion import merge_multiple_dataframe
from training import train_model
from config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

# Set up paths
input_folder_path = CONFIG['input_folder_path']
test_data_path = CONFIG['test_data_path']
prod_deployment_path = CONFIG['prod_deployment_path']
output_folder_path = CONFIG['output_folder_path']
final_data_path = os.path.join(
    os.getcwd(), 
    CONFIG["output_folder_path"], 
    'finaldata.csv'
)
test_data_csv_path = os.path.join(
    os.getcwd(), 
    CONFIG["test_data_path"], 
    'testdata.csv'
)

def model_predictions(test_data_csv_path, prod_deployment_path):
    """
    Load the trained model and make predictions on test data.

    Args:
        test_data_csv_path (str): Path to the test data CSV file.
        prod_deployment_path (str): Path to the production deployment folder.

    Returns:
        y_pred (list): List of predictions.
    """
    lr_model = pickle.load(
        open(
            os.path.join(
                os.getcwd(), 
                prod_deployment_path, 
                "trained_model.pkl"
            ), 
            'rb'
        )
    )
    test_df = pd.read_csv(test_data_csv_path)
    test_df = test_df.drop(['corporation'], axis=1)

    X_test = test_df.iloc[:, :-1]
    y_pred = lr_model.predict(X_test)
    return y_pred


def dataframe_summary(test_data_csv_path):
    """
    Calculate summary statistics of the test data.

    Args:
        test_data_csv_path (str): Path to the test data CSV file.

    Returns:
        summary_statistics (list): List of summary statistics.
    """
    final_df = pd.read_csv(test_data_csv_path)
    final_df = final_df.drop(['corporation'], axis=1)
    X = final_df.iloc[:, :-1]

    # Calculate summary statistics of the test data
    summary = X.describe().transpose()
    summary_statistics = summary[
        ['mean', '50%', 'std', 'min', 'max']
    ].values.flatten().tolist()
    return summary_statistics


def missing_data(test_data_csv_path):
    """
    Calculate the percentage of missing data in the test data.

    Args:
        test_data_csv_path (str): Path to the test data CSV file.

    Returns:
        missing_values (list): List of missing values.
    """
    final_df = pd.read_csv(test_data_csv_path)
    final_df = final_df.drop(['corporation'], axis=1)

    missing_values = final_df.isna().mean().tolist()
    return missing_values


def execution_time(input_folder_path, prod_deployment_path):
    """
    Calculate the average time taken for ingestion and training.

    Args:
        input_folder_path (str): Path to the input folder.
        prod_deployment_path (str): Path to the production deployment folder.

    Returns:
        ingestion_time (float): Average time taken for ingestion.
        training_time (float): Average time taken for training.
    """
    iteration = 10
    starttime = timeit.default_timer()
    ingested_files_path = os.path.join(
        os.getcwd(), 
        prod_deployment_path, 
        'ingestedfiles.txt'
    )
    for _ in range(iteration):
        merge_multiple_dataframe(
            input_folder_path, 
            ingested_files_path, 
            final_data_path
        )
    ingestion_time = (timeit.default_timer() - starttime) / iteration
    LOGGER.info('Ingestion time: {}'.format(ingestion_time))

    prod_model_train_path = os.path.join(
        os.getcwd(), 
        prod_deployment_path, 
        'trained_model.pkl'
    )
    starttime = timeit.default_timer()
    for _ in range(iteration):
        train_model(final_data_path, prod_model_train_path)
    training_time = (timeit.default_timer() - starttime) / iteration
    LOGGER.info('Training time: {}'.format(training_time))
    return ingestion_time, training_time


def outdated_packages_list():
    """
    Check for outdated packages in the requirements.txt file.

    Returns:
        outdated_packages (list): List of outdated packages.
    """
    df = pd.DataFrame(columns=['package_name', 'current', 'recent_available'])

    with open("requirements.txt", "r") as file:
        strings = file.readlines()
        package_names = []
        curent_versions = []
        recent = []

        for line in strings:
            package_name, cur_ver = line.strip().split('==')
            package_names.append(package_name)
            curent_versions.append(cur_ver)
            info = subprocess.check_output(
                ['python', '-m', 'pip', 'show', package_name])
            recent.append(str(info).split('\\n')[1].split()[1])

        df['package_name'] = package_names
        df['current'] = curent_versions
        df['recent_available'] = recent
    LOGGER.info('Outdated packages: {}'.format(df.values.tolist()))
    return df.values.tolist()



if __name__ == '__main__':
    model_predictions(test_data_csv_path, prod_deployment_path)
    dataframe_summary(test_data_csv_path)
    missing_data(test_data_csv_path)
    execution_time(input_folder_path, prod_deployment_path)
    outdated_packages_list()





    
