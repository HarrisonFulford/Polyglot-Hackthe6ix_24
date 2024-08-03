
# Imports the Google Cloud Translation library
"""from google.cloud import translate

# Initialize Translation client
def translate_text(
    text: str = "Orange", project_id: str = "involuted-span-430807-i9"
) -> translate.TranslationServiceClient:
    """"""Translating Text.""""""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "en-US",
            "target_language_code": "fr",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return response

def translate_text_with_model(
    text: str = "YOUR_TEXT_TO_TRANSLATE",
    project_id: str = "YOUR_PROJECT_ID",
    model_id: str = "YOUR_MODEL_ID",
) -> translate.TranslationServiceClient:
    """"""Translates a given text using Translation custom model.""""""

    client = translate.TranslationServiceClient()

    location = "us-central1"
    parent = f"projects/{project_id}/locations/{location}"
    model_path = f"{parent}/models/{model_id}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    response = client.translate_text(
        request={
            "contents": [text],
            "target_language_code": "ja",
            "model": model_path,
            "source_language_code": "en",
            "parent": parent,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
        }
    )
    # Display the translation for each input text provided
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return response"""


"""languages = ["fr", #French
        "es", #Spanish
        "pt", #Portuguese
        "hi", #Hindi
        "zh", #Chinese 
        "ar", #Arabic
        "bn", #Bengali
        "jp" #Japanese
        ]
languagesIndex = 4

objectName = "Apple"
to_lang = languages[languagesIndex]

translator= Translator(to_lang)
#Recive input from object reconision
translation = translator.translate("Apple")
print(translation)"""