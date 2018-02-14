import os

from lore.encoders import Token, Unique, Norm
import lore.io
import lore.pipelines
import lore.env

import pandas


class Holdout(lore.pipelines.holdout.Base):
    def get_data(self):
        # You can inspect the source data csv's yourself from the command line with:
        # $ wget https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz
        # $ tar -xzvf instacart_online_grocery_shopping_2017_05_01.tar.gz
        url = 'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'
        lore.io.download(url, cache=True, extract=True)
        
        # Defined to DRY up paths to 3rd party file hierarchy
        def read_csv(name):
            return pandas.read_csv(os.path.join(lore.env.data_dir, 'instacart_2017_05_01', name + '.csv'))
        
        # Published data was split into irrelavent sets, so we will combine them to re-purpose all the data.
        orders = read_csv('order_products__prior').append(read_csv('order_products__train'))
        products = read_csv('products').set_index('product_id')
        
        # count how many times each product was ordered
        sales = products.join(orders.groupby('product_id').size().to_frame('sales'))

        # add department names to the sales data
        departments = read_csv('departments').set_index('department_id')
        sales = sales.set_index('department_id').join(departments).reset_index()
        
        # Only return the columns we need for training
        return sales[['product_name', 'department', 'sales']]
    
    def get_encoders(self):
        return (
            # An encoder to tokenize product names into max 10 tokens that occur in the corpus at least 5 times.
            Token('product_name', sequence_length=10, minimum_occurrences=5),
            # An encoder to translate department names into unique identifiers that occur at least 50 times
            Unique('department', minimum_occurrences=50)
        )
    
    def get_output_encoder(self):
        # Sales is floating point which we could Pass encode, but Norm will bring it to small values around 0,
        # which are more amenable to deep learning.
        return Norm('sales')
