from app.models.datasets import DatasetModel
import pandas as pd
import numpy as np

# Sklearn modules for encoding, scaling and dimensionality reduction
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA

class DatasetPreprocessingForModel:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.dataset_model = DatasetModel()

    def fetch_dataset(self):
        # Retrieve dataset details and CSV file
        self.dataset = self.dataset_model.get_by_id(self.dataset_id)
        self.df = self.dataset_model.get_dataset_csv(self.dataset_id)
        # Expected datatypes provided by the user
        self.expected_datatypes = self.dataset.get("datatype_of_each_column", {})
        return self.df

    def convert_columns_to_datatype(self):
        """
        Convert dataframe columns to the types provided in expected_datatypes.
        Example expected_datatypes:
            {
                "Id": "uint",
                "SepalLengthCm": "float",
                "SepalWidthCm": "float",
                "PetalLengthCm": "float",
                "PetalWidthCm": "float",
                "Species": "category",
                "ObservationDate": "datetime"  # if provided
            }
        """
        for col, expected_type in self.expected_datatypes.items():
            if col in self.df.columns:
                etype = expected_type.lower()
                try:
                    if etype in ["float", "double"]:
                        self.df[col] = self.df[col].astype(float)
                    elif etype in ["int", "uint", "integer"]:
                        self.df[col] = self.df[col].astype(int)
                    elif etype in ["category"]:
                        self.df[col] = self.df[col].astype("category")
                    elif etype in ["datetime", "date", "time"]:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        # Extract parts if conversion is successful
                        self.df[f"{col}_day"] = self.df[col].dt.day
                        self.df[f"{col}_month"] = self.df[col].dt.month
                        self.df[f"{col}_year"] = self.df[col].dt.year
                        self.df[f"{col}_hour"] = self.df[col].dt.hour
                        self.df[f"{col}_minute"] = self.df[col].dt.minute
                        self.df[f"{col}_second"] = self.df[col].dt.second
                except Exception as e:
                    print(f"Conversion issue on column {col}: {e}")
        return self.df

    def combine_date_time_columns(self):
        """
        If exactly one date column and one time column are detected (based on column name),
        combine them into a single datetime column.
        """
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        time_cols = [col for col in self.df.columns if 'time' in col.lower()]
        # If exactly one date and one time column exist, combine them.
        if len(date_cols) == 1 and len(time_cols) == 1:
            combined = self.df[date_cols[0]].astype(str) + ' ' + self.df[time_cols[0]].astype(str)
            self.df['datetime_combined'] = pd.to_datetime(combined, errors='coerce')
            # Optionally, extract datetime parts for the combined column
            self.df["datetime_combined_day"] = self.df['datetime_combined'].dt.day
            self.df["datetime_combined_month"] = self.df['datetime_combined'].dt.month
            self.df["datetime_combined_year"] = self.df['datetime_combined'].dt.year
            self.df["datetime_combined_hour"] = self.df['datetime_combined'].dt.hour
            self.df["datetime_combined_minute"] = self.df['datetime_combined'].dt.minute
            self.df["datetime_combined_second"] = self.df['datetime_combined'].dt.second
        return self.df

    def separate_columns(self):
        """
        Identify numerical, categorical, and datetime columns.
        If target column is provided, remove it from features.
        """
        # Numerical columns (after conversion)
        self.numerical_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        # Categorical columns
        self.categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Datetime columns might have been converted to datetime64; we can list them if needed.
        self.datetime_cols = self.df.select_dtypes(include=["datetime"]).columns.tolist()
        
        # Exclude target column from features if provided
        self.target_column = self.dataset.get("target_column", "")
        if self.target_column:
            if self.target_column in self.categorical_cols:
                self.categorical_cols.remove(self.target_column)
            if self.target_column in self.numerical_cols:
                self.numerical_cols.remove(self.target_column)
            if self.target_column in self.datetime_cols:
                self.datetime_cols.remove(self.target_column)
                
        return self.numerical_cols, self.categorical_cols, self.datetime_cols

    def encode_columns(self):
        """
        Encode categorical features and, if necessary, the target column.
        """
        # One-hot encode categorical features
        if self.categorical_cols:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            cat_data = ohe.fit_transform(self.df[self.categorical_cols])
            cat_df = pd.DataFrame(cat_data, 
                                  columns=ohe.get_feature_names_out(self.categorical_cols),
                                  index=self.df.index)
            self.df.drop(columns=self.categorical_cols, inplace=True)
            self.df = pd.concat([self.df, cat_df], axis=1)

        # Encode target column if it's categorical
        if self.target_column and self.target_column in self.df.columns:
            if self.df[self.target_column].dtype.name == "category" or self.df[self.target_column].dtype == object:
                le = LabelEncoder()
                self.df[self.target_column] = le.fit_transform(self.df[self.target_column])
        return self.df

    def scale_numerical(self):
        """
        Apply scaling to numerical columns.
        Standardization is applied if flagged. Normalization is applied if flagged.
        """
        if self.numerical_cols:
            # Standardization
            if self.dataset.get("standardization_necessary", False):
                scaler = StandardScaler()
                self.df[self.numerical_cols] = scaler.fit_transform(self.df[self.numerical_cols])
            
            # Normalization (Min-Max scaling) may be applied additionally if needed.
            if self.dataset.get("normalization_necessary", False):
                scaler = MinMaxScaler()
                self.df[self.numerical_cols] = scaler.fit_transform(self.df[self.numerical_cols])
        return self.df

    def remove_highly_correlated_columns(self, threshold=0.95):
        """
        Remove one column from each pair of highly correlated numerical features.
        """
        if self.numerical_cols:
            corr_matrix = self.df[self.numerical_cols].corr().abs()
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find features with correlation greater than the threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            if to_drop:
                print(f"Dropping highly correlated columns: {to_drop}")
                self.df.drop(columns=to_drop, inplace=True)
                # Update numerical columns list after drop
                self.numerical_cols = [col for col in self.numerical_cols if col not in to_drop]
        return self.df

    def reduce_dimensionality(self, n_components=0.95):
        """
        Apply PCA on numerical features if dimensionality reduction is flagged.
        """
        if self.dataset.get("dimensionality_reduction", False) and self.numerical_cols:
            pca = PCA(n_components=n_components)
            numerical_data = self.df[self.numerical_cols]
            pca_transformed = pca.fit_transform(numerical_data)
            pca_df = pd.DataFrame(pca_transformed, 
                                  columns=[f"PC{i+1}" for i in range(pca_transformed.shape[1])],
                                  index=self.df.index)
            self.df.drop(columns=self.numerical_cols, inplace=True)
            self.df = pd.concat([self.df, pca_df], axis=1)
        return self.df

    def increase_dataset_size(self):
        """
        A simple oversampling example:
        If the flag is enabled, duplicate the dataset to increase its size.
        In practice, consider using SMOTE or other augmentation techniques.
        """
        if self.dataset.get("increase_the_size_of_dataset", False):
            print("Increasing dataset size by duplicating rows.")
            self.df = pd.concat([self.df, self.df.copy()], ignore_index=True)
        return self.df

    def split_train_test(self):
        """
        Split the dataset into training and testing sets.
        """
        test_percentage = self.dataset.get("test_dataset_percentage", 30) / 100.0
        n = len(self.df)
        test_size = int(n * test_percentage)
        train_df = self.df.iloc[:-test_size, :].reset_index(drop=True)
        test_df = self.df.iloc[-test_size:, :].reset_index(drop=True)
        return train_df, test_df

    def main(self):
        # Main processing pipeline
        self.fetch_dataset()
        self.convert_columns_to_datatype()
        self.combine_date_time_columns()
        self.separate_columns()
        self.encode_columns()
        self.scale_numerical()
        self.remove_highly_correlated_columns()
        self.reduce_dimensionality()
        self.increase_dataset_size()
        train_df, test_df = self.split_train_test()

        # Save processed dataset for future testing
        processed_filename = f"processed_{self.dataset.get('filename', 'dataset')}.csv"
        train_df.to_csv(processed_filename, index=False)
        print(f"Processed dataset saved as {processed_filename}")

        # Optionally, update dataset status in your database/model
        # self.dataset_model.update(self.dataset_id, {"is_preprocessing_done": True})
        return train_df, test_df

# Example usage:
if __name__ == "__main__":
    dataset_id = "your_dataset_id_here"
    processor = DatasetPreprocessingForModel(dataset_id)
    train, test = processor.main()
