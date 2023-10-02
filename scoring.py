import os
import pickle
import logging
import pandas as pd

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
output_folder_model_path = CONFIG['output_folder_path']
test_data_path =  os.path.join(
    os.getcwd(),
    CONFIG['test_data_path'],
    "trained_model.pkl"
)
train_model_path = os.path.join(
    os.getcwd(),
    output_folder_model_path,
    "trained_model.pkl"
)
test_df = pd.read_csv(test_data_path)

def score_model(
    train_model_path, 
    test_df
):
    """
    Score the trained model using test data and 
    save the score in the output folder.
    Args:
        train_model_path (str): Path to the trained model pickle file.
        test_df (pd.DataFrame): Test data.

    Returns:
        f1_score (float): F1 score of the model.
    """
    LOGGER.info('Scoring the model')
    # Load the trained model
    lr_model = pickle.load(open(train_model_path, 'rb'))
    # Prepare test data
    test_df = test_df.drop(["corporation"], axis=1)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    # Make predictions
    y_pred = lr_model.predict(X_test)
    # Calculate F1 score
    f1_score = metrics.f1_score(y_test, y_pred)
    # Save the score in the output folder
    latest_score_path = os.path.join(
        os.getcwd(), 
        train_model_path, 
        'latestscore.txt'
    )
    with open(latest_score_path, 'w') as f:
        f.write(str(f1_score))

    LOGGER.info('F1 score: {}'.format(f1_score))
    return f1_score

if __name__ == '__main__':
    score_model(
        train_model_path,
        test_df
    )