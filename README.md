**Real Estate Insights: Predictive Analytics for Airbnb and Zillow Listings**

**Motivation**

The swiftly changing financial environment has made it difficult to ensure financial stability as
individuals approach retirement. Real estate investment offers a compelling opportunity to
supplement traditional retirement plans such as 401(k)s and IRAs. By investing in real estate,
individuals can gain immediate rental income and benefit from long-term property appreciation,
providing a hedge against inflation. However, the challenge lies in identifying the best markets
and properties, and in devising effective pricing strategies. Real estate investment has gained
popularity as a strategy for supplemental retirement planning. Platforms like Airbnb allow
property owners to tap into short-term rental markets, while the potential for long-term capital
gains continues to be an attractive aspect of real estate investment. However, navigating the
complex housing market dynamics and making well-informed investment decisions requires
specialized knowledge and tools. This project aims to leverage extensive public listing data from
sources like Airbnb and Zillow, utilizing advanced machine learning and deep learning
techniques to develop predictive analytics models. These models will provide investors—both
individuals and real estate professionals—with refined insights into optimal markets, property
attributes, and pricing strategies tailored for maximizing returns from both short-term rentals and
long-term property investments. The goal is to enhance real estate-centered financial planning
and improve retirement readiness, offering investors increased control over their financial
futures.

Dataset for Airbnb Data: https://insideairbnb.com/about/

Dataset for Zillow Data: https://www.zillow.com/research/data/

**Data Preprocessing**

The project involves meticulous data preprocessing steps tailored to enhance the quality and
reliability of the analyses. This includes handling missing data using mean and mode imputation
for numerical data and integrating KNN imputation for potential future categorical nulls. Outliers
are detected and managed using the IQR method, with extreme values replaced by NaN and
filled with the median. Further transformations involve dropping redundant columns, adding new
ones, correcting data types, standardizing numerical features, and one-hot encoding categorical
features. Finally, the cleaned Airbnb and Zillow datasets are concatenated into a cohesive CSV
file for comprehensive analysis and model development.

**Modeling**

In the Airbnb and Zillow price prediction project, three models are used for Airbnb: Linear
Regression, Random Forest Regressor, and XGBoost. These models are fine-tuned to improve
accuracy. For Zillow, two advanced models, LSTM and N-BEATS, are used. Their performance
is evaluated using specific metrics to ensure reliable predictions.

**Steps to Run the Code**

Step 1: Download and install Visual Studio Code from the official website, following the installation instructions for your operating system.

Step 2: Ensure that Python is installed on your machine by downloading it from python.org. While installing, remember to select the option to "Add Python to PATH" if you are using Windows.

Step 3: Open Visual Studio Code, go to the Extensions view by clicking on the square icon on
the sidebar or pressing Ctrl+Shift+X, and search for "Python." Install the extension provided by
Microsoft for enhanced functionality with Python.

Step 4: Download the project or clone the repository in your local environment. Unzip the files if
downloaded as a zip file into your respective directory

Step 5: Navigate to the repository, open a new command prompt terminal, and type 'cd..'
Create a new virtual environment by typing 'python -m venv myenv'. The environment is outside
the repository to prevent redundant installations and keep each environment private to individual
users.

Step 6: Activate the created environment by passing the command '.\myenv\Scripts\activate' for
Windows and 'source myenv/bin/activate' for macOS/Linux.

Step 7: Once the environment is activated, navigate to the repository using 'cd code' and type
'pip install -r requirements.txt' to install all the dependencies within the environment

Step 8: Run the command 'python setup.py extract' to get all the data and store it within your
directory with a folder name 'data'

Step 9: Run 'python eda.py' to get all exploratory data analysis and visualization plots on both
Airbnb and Zillow data.

Step 10: Run 'python preprocess.py' to cleanse, preprocess, and store the processed data
within the environment. The Airbnb data is stored as 'all_airbnb_processed.csv' and Zillow data
is stored as 'all_zillow_processed.csv' within the 'data' directory.

Step 11: Run 'python model.py' to define all the base and hyper-tuned models for both Airbnb
and Zillow data

Step 12: Run 'python train.py' to train all those models with the preprocessed data and save it
within the environment.

Step 13: Run 'python test.py' to test the models on test data sets and evaluate the models
based on the evaluation metrics being displayed.

**Team Contribution**

Mahamaya, Aradhya : Data Collection & EDA, Airbnb Dataset

Shashi, Bhavan : Data Collection & EDA, Zillow Dataset

Mahamaya, Aradhya : Data Pre-processing, Airbnb Dataset

Shashi, Bhavan : Data Pre-processing, Zillow Dataset

Shashi, Bhavan : Model Development, Random Forest Regressor and XG Boost Regressor

Aradhya, Bhavan : Model Development, Linear Regression

Mahamaya, Aradhya : Model Development, N-BEATS and Long-Term Short Memory (LSTM)

Shashi, Mahamaya : Power BI Data Analysis, Zillow and Airbnb Dashboard

Aradhya, Bhavan : Power BI Data Analysis, Comparison, and Prediction Dashboard for Airbnb and Zillow Datasets