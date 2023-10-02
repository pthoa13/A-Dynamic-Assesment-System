import os
import logging
import pandas as pd
import seaborn as sns
from sklearn import metrics

from diagnostics import model_predictions
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

dataset_csv_path = CONFIG['output_folder_path']
output_model_path = CONFIG['output_model_path']
prod_deployment_path = CONFIG['prod_deployment_path']

test_data_csv_path = os.path.join(
    os.getcwd(), 
    CONFIG["test_data_path"], 
    'testdata.csv'
)
cfm_path = os.path.join(
    os.getcwd(),
    CONFIG["output_model_path"],
    'confusionmatrix.png'
)

def report_score_model(test_data_csv_path, cfm_path, prod_deployment_path):
    """
    Score the model on the test data and save the score in the output folder.

    Args:
        test_data_csv_path (str): Path to the test data CSV file.
        cfm_path (str): Path to the confusion matrix file.
        prod_deployment_path (str): Path to the production deployment folder.
    """
    test_data_csv = pd.read_csv(test_data_csv_path)
    y_true = test_data_csv['exited'].values
    y_pred = model_predictions(test_data_csv_path, prod_deployment_path)

    cfm = metrics.confusion_matrix(y_true, y_pred)
    LOGGER.info('Confusion matrix: {}'.format(cfm))

    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cfm, index=classes, columns=classes)
    cfm_plot = sns.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig(cfm_path)
    LOGGER.info('Confusion matrix saved at {}'.format(cfm_path))


if __name__ == '__main__':
    report_score_model(
        test_data_csv_path, 
        cfm_path, 
        prod_deployment_path
    )