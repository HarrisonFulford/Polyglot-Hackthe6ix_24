import os
import json
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_google_application_credentials():
    # Retrieve the JSON credentials from the environment variable
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if credentials_json is None:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is not set.")

    # Parse the JSON string into a dictionary
    credentials_dict = json.loads(credentials_json)
    
    # Save the credentials to a temporary file
    with open('temp_credentials.json', 'w') as temp_file:
        json.dump(credentials_dict, temp_file)
    
    return 'temp_credentials.json'

def translate_text(text, target_language='en'):
    # Get the path to the temporary credentials file
    credentials_path = get_google_application_credentials()
    
    # Instantiate a client using the temporary credentials
    translate_client = translate.Client.from_service_account_json(credentials_path)

    # Perform the translation
    result = translate_client.translate(text, target_language=target_language)
    
    # Optionally, delete the temporary file if no longer needed
    os.remove(credentials_path)
    
    return result['translatedText']

# Example usage
text_to_translate = "Bonjour tout le monde"
translated_text = translate_text(text_to_translate, target_language='zh')
print(translated_text)