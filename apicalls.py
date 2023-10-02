import os
import requests
import logging
from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)
URL = "http://127.0.0.1:8000"
CONFIG = get_config()
api_returns_path = os.path.join(
    os.getcwd(),
    CONFIG["output_model_path"],
    'apireturns.txt'
)

def request_api(URL, api_returns_path):
    """
    Make API calls to the deployed model

    Args:
        URL (str): URL of the deployed model
        api_returns_path (str): path to the file where the API returns are saved
    """
    LOGGER.info('Calling the API')
    response_prediction = requests.post(
        URL + '/prediction?input_data=testdata/testdata.csv').content
    response_scoring = requests.get(URL + '/scoring').content
    response_summarystats = requests.get(URL + '/summarystats').content
    response_diagnostics = requests.get(URL + '/diagnostics').content

    responses = [
        response_prediction,
        response_scoring,
        response_summarystats,
        response_diagnostics
    ]
    LOGGER.info('API responses: {}'.format(responses))

    with open(api_returns_path, 'w') as file:
        file.write(str(responses))
        
    LOGGER.info('API responses saved at {}'.format(api_returns_path))


if __name__ == '__main__':
    request_api(URL, api_returns_path)




