from googletrans import Translator
languages = ["fr", #French
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
print(translation)