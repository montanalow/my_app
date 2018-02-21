import lore.models.keras

import my_app.pipelines.product_popularity
import my_app.estimators.product_popularity


class Keras(lore.models.keras.Base):
    def __init__(self, pipeline=None, estimator=None):
        super(Keras, self).__init__(
            my_app.pipelines.product_popularity.Holdout(),
            my_app.estimators.product_popularity.Keras(
                hidden_layers=2,
                embed_size=4,
                hidden_width=256,
                batch_size=1024,
                sequence_embedding='lstm',
            )
        )
