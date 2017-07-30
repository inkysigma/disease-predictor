"""The linear trainer for data"""
import tensorflow as tf
import pandas as pd
import tempfile
from collector import *
from reader import construct

POLITICAL_PARTIES = ["independent",
                     "green", "democrat", "republican", "libertarian"]
DIRECTORY = None


class Predictor:
    def build_predictor(self):
        county = tf.contrib.layers.sparse_column_with_hash_bucket(
            "county", hash_bucket_size=1000)
        disease = tf.contrib.layers.sparse_column_with_hash_bucket(
            "disease", hash_bucket_size=1000
        )
        registered = tf.contrib.layers.real_valued_column("registered")
        political_parties = [(tf.contrib.layers.real_valued_column(party))
                             for party in POLITICAL_PARTIES]
        births = tf.contrib.layers.real_valued_column("births")
        fertility_rate = tf.contrib.layers.real_valued_column("fertility_rate")
        birth_rate = tf.contrib.layers.real_valued_column("birth_weight")
        mother_age = tf.contrib.layers.real_valued_column("mother_age")
        overall_poverty = tf.contrib.layers.real_valued_column(
            "overall_poverty"
        )
        population_density = tf.contrib.layers.real_valued_column(
            "population_density"
        )
        youth_poverty = tf.contrib.layers.real_valued_column("youth_poverty")
        median_income = tf.contrib.layers.real_valued_column("median_income")

        wide = [
            county, disease
        ]

        deep = [
            tf.contrib.layers.embedding_column(county, dimension=58),
            tf.contrib.layers.embedding_column(disease, dimension=65),
            registered,
            births,
            fertility_rate,
            birth_rate,
            mother_age,
            overall_poverty,
            youth_poverty,
            median_income
        ]
        deep = deep + political_parties

        model_dir = tempfile.mkdtemp()
        m = tf.contrib.learn.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            linear_feature_columns=wide,
            dnn_feature_columns=deep,
            dnn_hidden_units=[10, 4],
            fix_global_step_increment_bug=True,
            linear_optimizer=tf.train.FtrlOptimizer(
                learning_rate=0.0001,
                l1_regularization_strength=100.0,
                l2_regularization_strength=100.0),
            dnn_optimizer=tf.train.AdagradOptimizer(
                learning_rate=0.01,
                initial_accumulator_value=0.1
            ))

        return m

    def train(self, year):
        m = self.build_predictor()
        m.fit(input_fn=lambda: construct(
            "data/{0}merged.csv".format(year), year), steps=100)
        self.model = m

    def test(self):
        pass

    def predict(self, input_file):
        pass
