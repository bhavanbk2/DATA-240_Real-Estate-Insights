import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.airbnb_datasets = {}
        self.zillow_datasets = {}

    def extract_beds_from_filename(self, filename):
        """Extracts the number of beds from the filename assuming the format 'zillow_Xb.csv'."""
        try:
            # Split by '_', take the second part, remove 'b.csv' and convert to integer
            bed_part = filename.split('_')[1]  # Gets '1b.csv'
            bed_number = bed_part.replace('b.csv', '')  # Removes 'b.csv', resulting in '1'
            return int(bed_number)
        except (IndexError, ValueError) as e:
            print(f"Error processing filename '{filename}': {e}")
            return None  # Return None or a default value if parsing fails

    
    def load_data(self):
        """Load data from the specified directory, handling Airbnb and Zillow datasets separately."""
        data_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        for filename in data_files:
            dataset_name = os.path.splitext(filename)[0]
            file_path = os.path.join(self.data_path, filename)
            if 'zillow' in filename:
                beds = self.extract_beds_from_filename(filename)
                if beds is not None:
                    df = pd.read_csv(file_path)
                    df = self.transform_zillow(df, beds)
                    self.zillow_datasets[dataset_name] = df
            elif 'listings' in filename:
                city_name = filename.split('_')[0]  # Assumes the filename format `City_listings.csv`
                df = pd.read_csv(file_path)
                df = self.preprocess_airbnb(df, city_name)
                self.airbnb_datasets[dataset_name] = df

    def preprocess_airbnb(self, df, city_name):
        df = self.read_data(df)
        df = self.clean_airbnb(df)
        df = self.encode_and_impute(df)
        df = self.remove_outliers(df)
        df['city'] = city_name  # Add the city column
        return df

    def read_data(self, df):
        columns = ['id', 'neighbourhood_cleansed', 'room_type', 'amenities', 'accommodates', 'bathrooms', 'beds', 
                   'price', 'minimum_nights', 'availability_365', 'number_of_reviews', 'review_scores_rating', 
                   'review_scores_accuracy', 'review_scores_value']
        return df[columns]

    def clean_airbnb(self, df):
        df.loc[:, 'price'] = df['price'].str.replace(r'[^\d.]', '', regex=True).astype(float)
        df.loc[:, 'beds'] = df['beds'].fillna(0).astype(int)
        df.loc[:, 'amenities_count'] = df['amenities'].apply(lambda x: len(x.split(',')))
        df = df.drop(columns=['amenities', 'bathrooms'], errors='ignore')

        review_cols = ['minimum_nights', 'availability_365', 'number_of_reviews', 'review_scores_rating',
                   'review_scores_accuracy', 'review_scores_value']
        for col in review_cols:
            df.loc[:, col] = df[col].fillna(df[col].mean())
        return df

    def encode_and_impute(self, df):
        categorical_cols = ['neighbourhood_cleansed', 'room_type']
        encoders = {col: LabelEncoder().fit(df[col].dropna()) for col in categorical_cols}
        for col in categorical_cols:
            df[col] = df[col].fillna('Missing')
            df[col] = encoders[col].transform(df[col])

        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        df[categorical_cols] = pd.DataFrame(df[categorical_cols], columns=categorical_cols, index=df.index).round().astype(int)

        for col in categorical_cols:
            inv_map = {i: category for i, category in enumerate(encoders[col].classes_)}
            df[col] = df[col].map(inv_map)
        return df

    def remove_outliers(self, df):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for column in numerical_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    def transform_zillow(self, df, num_bed):
        df = self.clean_zillow(df)
        df['beds'] = num_bed
        return df

    def clean_zillow(self, df):
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for column in numeric_cols:
            median_val = df[column].median()
            df[column] = df[column].fillna(median_val)  # Corrected to avoid using inplace=True
        return df
    
def main():
    data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    processor = DataPreprocessor(data_path)
    processor.load_data()

    # Concatenate all Airbnb datasets into one DataFrame and save it
    if processor.airbnb_datasets:
        all_airbnb_data = pd.concat(processor.airbnb_datasets.values(), ignore_index=True)
        all_airbnb_data.to_csv(os.path.join(data_path, 'all_airbnb_processed.csv'), index=False)

    # Concatenate all Zillow datasets into one DataFrame and save it
    if processor.zillow_datasets:
        all_zillow_data = pd.concat(processor.zillow_datasets.values(), ignore_index=True)
        all_zillow_data.to_csv(os.path.join(data_path, 'all_zillow_processed.csv'), index=False)

if __name__ == '__main__':
    main()