import logging

import pandas as pd
from sklearn.externals import joblib


class TrainedModel(object):
    def __init__(self, trained_model_location):
        """

        :param trained_model_location: string that points to the location of the trained model metadata file

         This class is used to score arbitrary scikit-learn (known as sklearn) models. All sklearn models share
         share a paradigm of transform/fit/predict. This class requires the location of metadata and the predict
         method requires data in the form of a pandas dataframe in order to create predictions. Both probabilities
         and class predictions.
        """
        self.trained_model_location = trained_model_location

    def predict(self, prepared_data):
        """

        :param prepared_data:

        :return: pandas `DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>` including the model's score
        """

        loaded_model = joblib.load(self.trained_model_location)
        logging.info('model loaded from: %s' % self.trained_model_location)
        return_data = prepared_data.copy()
        scored_data = loaded_model.predict(prepared_data)
        try:
            return_data['prediction'] = scored_data
        except ValueError:
            return_data = pd.concat([return_data, scored_data], axis=1)

        if hasattr(loaded_model, 'predict_proba'):
            return_data['prediction_prob'] = loaded_model.predict_proba(prepared_data)[:, 1]
        else:
            return_data['prediction_prob'] = None
        return return_data


def load_score_model(model_path, prepared_data=None, **kwargs):
    trained_model = TrainedModel(model_path)
    scored_data = trained_model.predict(prepared_data)
    return scored_data
