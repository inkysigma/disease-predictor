import pandas as pd
import tensorflow as tf

CONTINUOUS_COLUMNS = ["births", "fertility_rate", "birth_weight", "mother_age",
                      "registered",	"democrat",	"republican", "independent",
                      "green", "libertarian", "median_income", "youth_poverty", "overall_poverty"]
CATEGORICAL_COLUMNS = ["county", "disease"]


def normalize(series):
    max = series.max()
    min = series.min()
    series = (series - min) / (max - min)
    series[series == 0] = 0.0001
    return series


def construct_features(df, diseases):
    df = merge_partitions(df, diseases)
    df.dropna(how="any")

    continuous = {
        k: tf.constant(normalize(pd.to_numeric(df[k].values)), dtype=float64)
        for k in CONTINUOUS_COLUMNS
    }
    categorical = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS
    }
    features = dict(continuous)
    features.update(categorical)

    print(features)
    input()
    return features


def construct_input(df, diseases):
    """A method for extrating data from the data the tensors required."""
    df = merge_partitions(df, diseases)
    df.dropna(how="any", axis=1)
    diseases.dropna(how="any", axis=1)
    for k in CONTINUOUS_COLUMNS:
        df[k] = normalize(pd.to_numeric(df[k].values))

    continuous = {
        k: tf.constant(df[k].values)
        for k in CONTINUOUS_COLUMNS
    }
    categorical = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS
    }
    labels = tf.constant(diseases["rate"].values)
    features = dict(continuous)
    features.update(categorical)
    return features, labels


def read_training(file):
    names = CONTINUOUS_COLUMNS.copy() + CATEGORICAL_COLUMNS
    return pd.read_csv(file)


def read_disease(year):
    frames = pd.read_csv("disease/cleandisease{0}.csv".format(year), names=[
                         "disease", "county", "count", "population", "rate"], skiprows=1)
    frames = frames[frames["disease"] == "Chlamydia"]
    return frames


def filter_counties(data: pd.DataFrame, frames):
    county = data["county"].values
    frames = frames[frames["county"].isin(county)]
    county = frames["county"].values
    frames = frames[frames["county"].isin(county)]
    return frames


def merge_partitions(data: pd.DataFrame, disease: pd.DataFrame):
    data = data.merge(disease, how="outer",
                      left_on="county", right_on="county")
    data.rename(columns={'disease_x': 'disease'}, inplace=True)
    return data


def read_features(file):
    return pd.read_csv(file)
