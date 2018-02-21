import unittest

import my_app.models.product_popularity

class TestProductPopularity(unittest.TestCase):
    def test_init(self):
        
        model = my_app.models.product_popularity.Keras()
        self.assertIsNotNone(model)

    def test_fit(self):
        
        model = my_app.models.product_popularity.Keras()
        model.pipeline.subsample = 100
        model.fit(epochs=1)
        self.assertTrue(True)

