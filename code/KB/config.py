import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

PROLOG_FILE = os.path.join(BASE_DIR, "facts_and_rules.pl")
CATEGORICAL_CSV = os.path.join(ROOT_DIR, "dataset", "diamonds_categorical.csv")
PREPROCESSED_CSV = os.path.join(ROOT_DIR, "dataset", "diamonds_preprocessed.csv")

TEST_OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output")
MODEL_PATH = os.path.join(TEST_OUTPUT_DIR, "model_path.joblib")
MINIKB_PATH = os.path.join(TEST_OUTPUT_DIR, "rules.json")
EXKB_PATH = os.path.join(TEST_OUTPUT_DIR, "composite_rules.json")
RANDOM_DIAMOND = os.path.join(TEST_OUTPUT_DIR, "random_diamond.json")

TARGET_COL = 'price'
SORTED_COLS = ['carat','cut','color','clarity','depth','table','x','y','z','price']
RANDOM_STATE = 42
CV_SPLITS = 5
