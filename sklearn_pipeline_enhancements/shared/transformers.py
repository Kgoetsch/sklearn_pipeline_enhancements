import numpy as np
import pandas as pd
from patsy.highlevel import dmatrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import _name_estimators, Pipeline

__author__ = 'kgoetsch'


def make_dataframeunion(steps):
    return DataFrameUnion(_name_estimators(steps))


class FactorExtractor(TransformerMixin, BaseEstimator):
    """
    In: pd.DataFrame
        Column in that Frame

    Out: pd.Series
    """

    def __init__(self, factor):
        self.factor = factor

    def transform(self, data):
        return data[self.factor]

    def fit(self, *_):
        return self


class RenameField(TransformerMixin, BaseEstimator):
    """
    In: pd.DataFrame
        Column in that Frame

    Out: pd.Series
    """

    def __init__(self, new_name):
        self.new_name = new_name

    def transform(self, data):
        data.name = self.new_name
        return data

    def fit(self, *_):
        return self


class FillNA(TransformerMixin, BaseEstimator):
    """
    In: pd.Series

    Out: pd.Series
    """

    def __init__(self, na_replacement=None):
        if na_replacement is not None:
            self.NA_replacement = na_replacement
        else:
            self.NA_replacement = 'missing'

    def transform(self, data):
        return data.fillna(self.NA_replacement)

    def fit(self, *_):
        return self


class DataFrameUnion(TransformerMixin, BaseEstimator):
    """
    In: list of (string, transformer) tuples :

    Out: pd.DataFrame
    """

    def __init__(self, transformer_list):
        self.feature_names = None
        self.transformer_list = transformer_list  # (string, Transformer)-tuple list

    def __getitem__(self, attrib):
        return self.__dict__[attrib]

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = (self._transform_one(trans, X)
              for name, trans in self.transformer_list)
        df_merged_result = self._merge_results(Xs)
        return df_merged_result

    def fit(self, X, y=None):
        """Fit all transformers using X.
        Parameters
        ----------
        :param X: pd.DataFrame
            Input data, used to fit transformers.

        :param y:
        """
        transformers = (self._fit_one_transformer(trans, X, y)
                        for name, trans in self.transformer_list)
        self._update_transformer_list(transformers)
        return self

    def _merge_results(self, transformed_result_generator):
        df_merged_result = ''
        for transformed in transformed_result_generator:
            if isinstance(transformed, pd.Series):
                transformed = pd.DataFrame(data=transformed)
            if not isinstance(df_merged_result, pd.DataFrame):
                df_merged_result = transformed
            else:
                df_merged_result = pd.concat([df_merged_result, transformed], axis=1)

        if self.feature_names is None:
            self.feature_names = df_merged_result.columns
        elif (len(self.feature_names) != len(df_merged_result.columns)) or \
                ((self.feature_names != df_merged_result.columns).any()):
            custom_dataframe = pd.DataFrame(data=0, columns=self.feature_names, index=df_merged_result.index)
            custom_dataframe.update(df_merged_result)
            df_merged_result = custom_dataframe
        return df_merged_result

    def _update_transformer_list(self, transformers):
        self.transformer_list[:] = [
            (name, new)
            for ((name, old), new) in zip(self.transformer_list, transformers)
            ]

    def _fit_one_transformer(self, transformer, X, y):
        return transformer.fit(X, y)

    def _transform_one(self, transformer, X):
        return transformer.transform(X)


def extract_and_denull(var, na=0):
    return Pipeline([
        ('extract', FactorExtractor(var)),
        ('fill_na', FillNA(na))
    ])


class ConvertToArray(TransformerMixin, BaseEstimator):
    """
    In: pd.Dataframe

    Out: np.array
    """

    def transform(self, data):
        return np.ascontiguousarray(data.values)

    def fit(self, *_):
        return self


class CategoricalDummifier(TransformerMixin, BaseEstimator):
    """
    In: pd.Series

    Out: pd.DataFrame
    """

    def transform(self, data):
        return dmatrix(formula_like=str(data.name), data=pd.DataFrame(data.apply(str)), return_type='dataframe',
                       NA_action='raise').drop('Intercept', axis=1)

    def fit(self, *_):
        return self


class WeekdayExtraction(TransformerMixin, BaseEstimator):
    """
    In: pd.DataFrame

    Out: pd.Series
    """

    def transform(self, data):
        return_data = pd.Series(data.index.weekday, index=data.index, name='weekday')
        return return_data

    def fit(self, *_):
        return self


class LengthofField(TransformerMixin, BaseEstimator):
    """
    In: pd.Series

    Out: pd.Series
    """

    def transform(self, data):
        return_value = data.apply(len)
        return return_value

    def fit(self, *_):
        return self


class InsertIntercept(TransformerMixin, BaseEstimator):
    """
    In: pd.DataFrame

    Out: pd.Series = 1 of the same length
    """

    def transform(self, data):
        return pd.DataFrame(data=1, index=data.index, columns=['Intercept'])

    def fit(self, *_):
        return self


if __name__ == '__main__':
    target = make_dataframeunion([extract_and_denull('years'), extract_and_denull('kitten')])
    for step in target.transformer_list:
        print step
