import os
import pickle
import logging
import pandas as pd

from config import get_config

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
                    
LOGGER = logging.getLogger(__name__)
CONFIG = get_config()

input_data_path = CONFIG['output_folder_path']
output_folder_model_path = CONFIG['output_model_path']
train_model_path = os.path.join(
    os.getcwd(),
    output_folder_model_path,
    "trained_model.pkl"
)

def train_model(
    input_data_path,
    output_model_path
):
    """
    Train the model using the provided dataset CSV file and 
    save it in the designated output model folder.

    Args:
        input_data_path (str): Path to the dataset CSV file.
        output_model_path (str): Path to the output model folder.
    """
    
    LOGGER.info("Training Model")
    # Model for training
    lr_model = LogisticRegression(
        C=1.0, 
        class_weight=None, 
        dual=False, 
        fit_intercept=True,
        intercept_scaling=1, 
        l1_ratio=None, 
        max_iter=100,
        multi_class='warn', 
        n_jobs=None, 
        penalty='l2',
        random_state=0, 
        solver='liblinear', 
        tol=0.0001, 
        verbose=0,
        warm_start=False
    )
    # Load processed data
    processed_df = pd.read_csv(input_data_path)
    processed_df = processed_df.drop(["corporation"], axis=1)
    # Create X, y
    X = processed_df.iloc[:, :-1]
    y = processed_df.iloc[:, -1]
    # Train test split
    X_train, _, y_train, _ = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=0
    )
    # Fit data to model
    lr_model.fit(X_train, y_train)
    # Save trained model
    with open(output_model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    LOGGER.info(f"Model saved at {output_model_path}")

    return output_model_path 

if __name__ == "__main__":
    train_model(
        input_data_path,
        train_model_path
    )