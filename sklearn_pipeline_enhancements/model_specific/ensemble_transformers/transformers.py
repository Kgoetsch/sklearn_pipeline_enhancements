import pandas as pd
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator


class predict_one_class(TransformerMixin, BaseEstimator):
    def __init__(self, model, y_variable):
        self.model = model
        self.y_variable = y_variable

    def fit(self, *data_args, **kwargs):
        X = data_args[0]
        y = data_args[1].apply(lambda x: 1 if x == self.y_variable else 0)
        self.model.fit(X, y)
        return self

    def transform(self, data, **transform_params):
        return pd.DataFrame(self.model.predict_proba(data)[:, 1],
                            columns=['cluster_' + str(self.y_variable) + '_prediction'])


class model_max(RegressorMixin, BaseEstimator):
    def predict(self, data):
        # grab just the cluster number from the column name for the max
        data['ensemble_prediction'] = data.idxmax(axis=1).apply(lambda x: x[8:9])
        return data

    def transform(self, data, *some_args):
        return data

    def fit(self, *args, **kwargs):
        return self
