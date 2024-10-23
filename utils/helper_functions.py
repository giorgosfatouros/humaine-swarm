import logging
from typing import Optional, Dict
# import pycountry    
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt
from chainlit import User
from typing import List, Any
from tools.functions_definition import functions
import json
from literalai import AsyncLiteralClient

lai = AsyncLiteralClient()



def setup_logging(logger_name, level=logging.INFO, log_file='error.log'):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:  # Check if the logger already has handlers
        # Create a console handler and set the level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

        # Create a file handler for logging errors to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger

logger = setup_logging('HELPER-FUNCTIONS', level=logging.ERROR)


def decode_jwt(token: str) -> Optional[Dict]:
    import jwt, os
    # Load the secret key securely from environment variables
    SECRET_KEY = os.getenv("CHAINLIT_AUTH_SECRET")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("JWT token has expired")
    except jwt.InvalidTokenError:
        logger.error("Invalid JWT token")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None


def extract_token_from_headers(headers: Dict) -> Optional[str]:
    auth_header = headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    referer = headers.get("referer")
    if referer and "=" in referer:
        return referer.split("=")[1].split('&')[0]
    return None


def extract_user_from_payload(payload: Dict) -> Optional[User]:
    user_id = payload.get("userId")
    user_email = payload.get("email")
    user_name = payload.get("name", '')
    token_provider = payload.get("provider", 'wix')
    plan = payload.get("plan", 'free')

    if not user_id or not user_email:
        return None

    role = "admin" if user_email in ["george@alpha-tensor.ai", "kostas@alpha-tensor.ai"] else "user"
    metadata = {"id": user_id, "email": user_email, "name": user_name, "plan": plan, "role": role, "provider": token_provider}
    return User(identifier=user_email, metadata=metadata)



def get_required_arguments(function_name: str):
    for func in functions:
        if func['name'] == function_name:
            required_params = func['parameters'].get('required', [])
            return required_params
    return []

async def get_literal_participant_id(user_id):
    if user_id:
        lai_user = await lai.api.get_or_create_user(identifier=user_id)
        return lai_user.id
    else:
        return None

def manage_chat_history(message_history):      
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    while total_length > 100000:
        if len(message_history) > 2:
            removed_msg = message_history.pop(2)  # Remove the third message (index 2)
            total_length -= len(json.dumps(removed_msg))
            logging.info(f"Reducing conversation history. Current length: {total_length} characters")
        else:
            break  # If we only have two or fewer messages, stop removing
    
    return message_history
# def get_country_code(country_name):
#     """
#     Get the Alpha-3 country code for a given country name.

#     Args:
#         country_name (str): The name of the country.

#     Returns:
#         str: The Alpha-3 country code if found, otherwise None.
#     """
#     try:
#         country = pycountry.countries.search_fuzzy(country_name)[0]
#         return country.alpha_3
#     except:
#         return None
    

def fetch_eod_macro_indicators(mapped_indicators, country_code):
    """
    Fetch data from EOD API for given indicators and country code.

    Args:
        mapped_indicators (list): List of indicators to fetch.
        country_code (str): The Alpha-3 country code.

    Returns:
        dict: A dictionary with indicators as keys and dataframes as values.
    """
    dataframes = {}
    
    for indicator in mapped_indicators:
        url = f'https://eodhd.com/api/macro-indicator/{country_code}?indicator={indicator}&api_token={os.getenv("EOD_API_TOKEN")}&fmt=json'
        response = requests.get(url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch data for {indicator}. Status code: {response.status_code}")
            continue
        
        data = response.json()
        # Convert data to dataframe
        df = pd.DataFrame(data)
        
        # Drop 'CountryName' and 'Period' columns if they exist
        df.drop(columns=['CountryName', 'Period'], errors='ignore', inplace=True)
        
        # Preprocess the dataframe to keep the data as timeseries
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep only the 15 latest dates
        df = df.tail(15)
        
        dataframes[indicator] = df.to_dict(orient='index')

    logger.info(dataframes)
    
    return dataframes



def create_plot(data_dict):
    """
    Create a plot for the given data dictionary.

    Args:
        data_dict (dict): A dictionary where keys are indicators and values are dictionaries with dates as keys and values as data points.

    Returns:
        matplotlib.figure.Figure: The created plot figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))  # Make the figure larger
    for indicator, data in data_dict.items():
        dates = list(data.keys())
        values = [entry['Value'] for entry in data.values()]
        ax.plot(dates, values, label=indicator)

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True)

    # # Set y-axis to log scale to ensure the scale of each plot doesn't affect the visualization of the other
    # if len(data_dict) > 1:
    #     ax.set_yscale('log')

    # Improve date visualization on x-axis
    fig.autofmt_xdate()
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-axis labels

    return fig