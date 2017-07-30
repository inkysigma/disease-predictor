"""The module for reading data"""
from collector import read_training, read_disease, filter_counties, construct_input

def construct(file, year):
    file = read_training(file)
    disease = read_disease(year)
    disease = filter_counties(file, disease)
    tensor = construct_input(file, disease)
    return tensor
