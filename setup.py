from setuptools import setup, Command
import os
import gdown

class ExtractCommand(Command):
    description = 'Download and prepare the dataset for analysis'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        file_ids = {
            'zillow_1b.csv': "1rdgiqbMLBWOhjQ6yHfUK249OfR8wM80k",
            'zillow_2b.csv': "1VVeljyc5NCMwW9WaLOMHoKfNWbEQwqRm",
            'zillow_3b.csv': "1T4MH6fPzjzzxaeN7ucPDIeswwmhEHzgl",
            'zillow_4b.csv': "1AmcIgfHgwMNjpDLaQ-2cvcr-RE0eWvnQ",
            'zillow_5b.csv': "1zuNwh6bSeuJnqnBNCs1rwo0yIJqXSClq",
            'SanFrancisco_listings.csv': "1F4S88V2rmSM0xK8NJAHuxP6tLn-2O73y",
            'Portland_listings.csv': "1VwFcHzWLXuapzfotSpHtthAL5lWzv9qC",
            'NewYork_listings.csv': "18nTcpGj1AVB2li5nbWuecSNBVcsWl3er",
            'NewOrleans_listings.csv': "1q_1pTQbsy28aiaGHPGF4XDoUh3MsCZHk",
            'Seattle_listings.csv': "1jDOrPAZGT6j6_IwHm7qegRg0vMaEeiZp",
            'SanDiego_listings.csv': "1qJmQwc8hzgMGy8NFM6Zfi3plmzmkHSC2",
            'LosAngeles_listings.csv': "10BjAuNEI46xBLcFncM0cjuQfUsROAN2i",
            'Denver_listings.csv': "1eIDZflCvayCz3uSjFeMAfoc-QTdz4lqU",
            'Boston_listings.csv': "1Dw4L3U-qsxuJYw0YM1kVL6-69xQot623"
        }
        base_url = 'https://drive.google.com/uc?id='
        
        my_path = os.path.abspath(os.path.dirname(__file__))
        os.makedirs(os.path.join(my_path, 'data'), exist_ok=True)

        for filename, file_id in file_ids.items():
            file_path = os.path.join(my_path, 'data', filename)
            file_url = f'{base_url}{file_id}'
            gdown.download(file_url, file_path, quiet=False)

setup(
    name='Airbnb-vs-Zillow-Price-Prediction',
    version='1.0',
    packages=['source'],
    url='https://github.com/AradhyaAlva/Data_Mining-Airbnb-vs-Zillow-Data-Prediction',
    license='free',
    author='Aradhya Alva Rathnakar',
    author_email='aradhyaalva15@gmail.com',
    description='A Data Mining project that aims to predict Airbnb and Zillow data to compare and analyze them side by side',
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines() if line.strip() and not line.startswith('#')
    ],
    cmdclass={
        'extract': ExtractCommand,
    }
)
