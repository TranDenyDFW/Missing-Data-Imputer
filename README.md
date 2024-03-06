# Missing-Data-Imputer

## Description
This project imputes missing information for a dataset. It can serve as a useful starting template for preprocessing Machine Learning data.

## Installation
No special installation steps required. Standard Python libraries used: json, pandas, numpy, sklearn.

## Usage
The script reads a CSV file into a pandas DataFrame, checks for columns with missing values, and then handles those missing values using various methods. The methods include mean, median, most_frequent, constant, knn, iterative, missing_indicator, remove, and categorical. The method to be used can be specified for each column.

Here is a basic usage example:

```python
import pandas as pd
df = pd.read_csv('data.csv')
df = handle_missing_data(df, columns_with_missing_values(df))
print(df)

