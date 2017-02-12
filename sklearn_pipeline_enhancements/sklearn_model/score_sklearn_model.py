import argparse
import logging
import pandas as pd
import sys

from sklearn_pipeline_enhancements.shared.utility_functions import download_sklearn_model
from sklearn_pipeline_enhancements.sklearn_model.TrainedModel import load_score_model

# set up logger to print to screen and write to log
root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


def score_sklearn_model(model_path, prepared_data_path, s3_bucket_input=None):
    """

    :param model_path: path to model on s3
    :param prepared_data_path: path to prepared data - just <schema>.<table_name> when pulling from hive metastore
    :param s3_bucket_input: s3 bucket that model lives in

    """
    if s3_bucket_input is not None:
        # Download Model
        model_path = download_sklearn_model(model_path, s3_bucket_input)

    prepared_data = pd.read_csv(prepared_data_path)
    logging.info('Prepared data loaded from from: %s' % prepared_data_path)

    scored_data = load_score_model(model_path,
                                   prepared_data=prepared_data)
    return scored_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--prepared_data_path")
    parser.add_argument("--s3_bucket_input")

    args = parser.parse_args()

    scored_data = score_sklearn_model(model_path=args.model_path, prepared_data_path=args.prepared_data_path,
                                      s3_bucket_input=args.s3_bucket_input)
