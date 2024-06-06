import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.airbnb_datasets = {}
        self.zillow_datasets = {}
        self.load_data()

    def load_data(self):
        data_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]

        for filename in data_files:
            dataset_name = os.path.splitext(filename)[0]
            if dataset_name.startswith('zillow'):
                self.zillow_datasets[dataset_name] = pd.read_csv(os.path.join(self.data_path, filename))
            elif 'listings' in dataset_name:
                self.airbnb_datasets[dataset_name] = pd.read_csv(os.path.join(self.data_path, filename))

    def plot_airbnb_price_line(self):
        for name, data in self.airbnb_datasets.items():
            if 'price' in data.columns and pd.api.types.is_numeric_dtype(data['price']):
                data['price'].plot(kind='line', figsize=(8, 4), title=f'Airbnb Price - {name}')
                plt.gca().spines[['top', 'right']].set_visible(False)
                plt.show()

    def plot_airbnb_room_type_distribution(self):
        for name, data in self.airbnb_datasets.items():
            if 'beds' in data.columns and 'room_type' in data.columns and pd.api.types.is_numeric_dtype(data['beds']):
                figsize = (12, 1.2 * len(data['room_type'].unique()))
                plt.figure(figsize=figsize)
                sns.violinplot(data=data, x='beds', y='room_type', inner='box', palette='Dark2')
                plt.title(f'Airbnb Room Type Distribution - {name}')
                sns.despine(top=True, right=True, bottom=True, left=True)
                plt.show()

    def plot_price_distribution_by_property_type(self):
        for name, data in self.airbnb_datasets.items():
            if 'price' in data.columns and 'property_type' in data.columns and pd.api.types.is_numeric_dtype(data['price']):
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=data, x='property_type', y='price')
                plt.title(f'Airbnb Price Distribution by Property Type - {name}')
                plt.xlabel('Property Type')
                plt.ylabel('Price')
                plt.xticks(rotation=45)
                plt.show()

    def plot_zillow_histograms(self):
        for name, data in self.zillow_datasets.items():
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
            if not numerical_features.empty:
                n_features = len(numerical_features)
                rows = (n_features // 4) + (n_features % 4 > 0)

                for i in range(0, n_features, 4):
                    fig, axs = plt.subplots(nrows=1, ncols=min(4, n_features-i), figsize=(20, 5))
                    for col, ax in zip(numerical_features[i:i+4], axs.flatten()):
                        data[col].hist(bins=15, ax=ax)
                        ax.set_title(f'{col} - {name}')
                    plt.tight_layout()
                    plt.show()

    def plot_zillow_boxplots(self):
        for name, data in self.zillow_datasets.items():
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
            if not numerical_features.empty:
                n_features = len(numerical_features)

                for i in range(0, n_features, 4):
                    fig, axs = plt.subplots(nrows=1, ncols=min(4, n_features-i), figsize=(20, 5))
                    for col, ax in zip(numerical_features[i:i+4], axs.flatten()):
                        sns.boxplot(y=data[col], ax=ax)
                        ax.set_title(f'{col} - {name}')
                    plt.tight_layout()
                    plt.show()

    def plot_zillow_region_count(self):
        for name, data in self.zillow_datasets.items():
            if 'RegionName' in data.columns and pd.api.types.is_numeric_dtype(data['RegionName']):
                plt.figure(figsize=(12, 8))
                sns.countplot(data=data, y='RegionName')
                plt.title(f'Zillow Region Count - {name}')
                plt.xlabel('Count')
                plt.ylabel('Region Name')
                plt.show()

    def plot_zillow_region_distribution_by_county(self):
        for name, data in self.zillow_datasets.items():
            if 'RegionName' in data.columns and 'CountyName' in data.columns and pd.api.types.is_numeric_dtype(data['RegionName']):
                max_unique_regions = 20
                max_unique_counties = 10

                top_regions = data['RegionName'].value_counts().nlargest(max_unique_regions).index
                top_counties = data['CountyName'].value_counts().nlargest(max_unique_counties).index

                filtered_data = data[(data['RegionName'].isin(top_regions)) & (data['CountyName'].isin(top_counties))]

                plt.figure(figsize=(12, 8))
                sns.violinplot(data=filtered_data, x='CountyName', y='RegionName')
                plt.title(f'Zillow Region Distribution by County - {name}')
                plt.xlabel('County Name')
                plt.ylabel('Region Name')
                plt.xticks(rotation=45)
                plt.show()


def main():
    # Define the path to the data directory based on setup.py configuration
    my_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(my_path, 'data')

    # Initialize DataVisualizer with the datasets in the data directory
    visualizer = DataVisualizer(data_path)

    # Perform Airbnb data visualizations
    visualizer.plot_airbnb_price_line()
    visualizer.plot_airbnb_room_type_distribution()
    visualizer.plot_price_distribution_by_property_type()

    # Perform Zillow data visualizations
    visualizer.plot_zillow_histograms()
    visualizer.plot_zillow_boxplots()
    visualizer.plot_zillow_region_count()
    visualizer.plot_zillow_region_distribution_by_county()


if __name__ == '__main__':
    main()
