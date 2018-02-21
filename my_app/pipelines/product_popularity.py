import os

from lore.encoders import Token, Unique, Norm
import lore.io
import lore.pipelines
import lore.env

import pandas


class Holdout(lore.pipelines.holdout.Base):
    # You can inspect the source data csv's yourself from the command line with:
    # $ wget https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz
    # $ tar -xzvf instacart_online_grocery_shopping_2017_05_01.tar.gz

    def get_data(self):
        url = 'https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz'

        # Lore will extract and cache files in lore.env.data_dir by default
        lore.io.download(url, cache=True, extract=True)
        
        # Defined to DRY up paths to 3rd party file hierarchy
        def read_csv(name):
            path = os.path.join(
                lore.env.data_dir,
                'instacart_2017_05_01',
                name + '.csv')
            return pandas.read_csv(path)
        
        # Published orders were split into irrelevant prior/train
        # sets, so we will combine them to re-purpose all the data.
        orders = read_csv('order_products__prior')
        orders = orders.append(read_csv('order_products__train'))

        # count how many times each product_id was ordered
        data = orders.groupby('product_id').size().to_frame('sales')

        # add product names and department ids to ordered product ids
        products = read_csv('products').set_index('product_id')
        data = data.join(products)

        # add department names to the department ids
        departments = read_csv('departments').set_index('department_id')
        data = data.set_index('department_id').join(departments)
        
        # Only return the columns we need for training
        data = data.reset_index()
        return data[['product_name', 'department', 'sales']]

    def get_encoders(self):
        return (
            # An encoder to tokenize product names into max 15 tokens that
            # occur in the corpus at least 10 times in the corpus. We also
            # want the estimator to spend 5x as many resources on name vs
            # department since there are so many more words in english
            # than their are grocery store departments.
            Token('product_name', sequence_length=15, minimum_occurrences=10, embed_scale=5),
            # An encoder to translate department names into unique
            # identifiers that occur at least 50 times
            Unique('department', minimum_occurrences=50)
        )

    def get_output_encoder(self):
        # Sales is floating point which we could Pass encode directly to the
        # estimator, but Norm will bring it to small values around 0,
        # which are more amenable to deep learning.
        return Norm('sales')
