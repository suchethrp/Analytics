# -*- coding: utf-8 -*-

# using pandas library to read csv or any delimited file and convert it to DataFrame for processing
import pandas as pd

inputFilePath = '~/Downloads/train.csv'

input_df = pd.read_csv(
        inputFilePath,
        sep='\t',                                 # delimiter
        usecols=['train_id', 'price'],            # columns name
        nrows=100,                                # number of rows
)

input_df.set_index('train_id', inplace=True)      # set index for DataFrame

# Manupulate the DataFrame by mapping through each columns

# function to check if the price is > 10 , then set flag
def price_flag(price):
    flag = 0
    if price > 10:
        flag = 1
    return flag

output_df = input_df
output_df['discounted_price'] = output_df['price'].map(lambda x : x * 0.85) # Apply 15 % discount on discounted_price

output_df['price_flag'] = output_df['price'].map(price_flag)

output_df[['discounted_price', 'price_flag']].to_csv('output.csv', index=True) # select the columns from DataFrame for output file
