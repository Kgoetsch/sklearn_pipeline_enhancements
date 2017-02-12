import numpy as np
import pandas as pd
from patsy.highlevel import dmatrix
from sklearn.base import TransformerMixin, RegressorMixin


class xgb_ModelTransformer(TransformerMixin):
    def __init__(self, name, model, eval_metric=None):
        self.name = name
        self.model = model
        self.eval_metric = eval_metric

    def fit(self, *args, **kwargs):
        self.model.fit(eval_metric=self.eval_metric, *args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X), columns=[self.name + '_prediction'])


class xgb_y_log_transform_ModelTransformer(TransformerMixin):
    def __init__(self, name, model, eval_metric=None):
        self.name = name
        self.model = model
        self.eval_metric = eval_metric

    def fit(self, *data_args, **kwargs):
        # bit hacky but functional
        assert (len(data_args) == 2)
        X = data_args[0]
        y = np.log(data_args[1])
        self.model.fit(X, y, eval_metric=self.eval_metric)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(np.e ** self.model.predict(X), columns=[self.name + '_prediction'])


class patsy_formula_transformer(TransformerMixin):
    """
    In: pd.Series

    Out: pd.Series
    """

    def __init__(self, formula, template_data):
        self.formula = formula
        self.template_data = template_data.copy()

    def transform(self, data):
        df_full = self.template_data
        df_new = data.copy()
        df_patsy = pd.concat([df_full, df_new])
        df_transformed = dmatrix(formula_like=self.formula, data=df_patsy, return_type='dataframe', NA_action='raise')
        df_return_data = df_transformed[-len(df_new):]

        return df_return_data

    def fit(self, *_):
        return self


class patsy_numeric_transformer(TransformerMixin):
    """
    In: pd.Series

    Out: pd.Series
    """

    def __init__(self, formula):
        self.formula = '+'.join(['0', formula])

    def transform(self, data):
        '''
        First time use reduced rank  transformer.
        Second plus times, use full rank transformer. The dataframe union that contains this transformer will
        automagically merge down to the same reduced rank
        :param data:

        '''

        return_data = dmatrix(formula_like=self.formula, data=data, return_type='dataframe', NA_action='raise')
        return return_data

    def fit(self, *_):
        return self


class patsy_cat_transformer(TransformerMixin):
    """
    EXPERIMENTAL - not in use

    In: pd.Series

    Out: pd.Series
    """

    def __init__(self, formula):
        self.formula = '+'.join(['0', formula])
        self.reference_column = None

    def transform(self, data):
        '''
        First time use reduced rank  transformer.
        Second plus times, use full rank transformer. The dataframe union that contains this transformer will
        automagically merge down to the same reduced rank
        :param data:

        '''

        return_data = dmatrix(formula_like=self.formula, data=data, return_type='dataframe', NA_action='raise')

        if self.reference_column is None:
            self.reference_column = return_data.columns[0]
        try:
            return_data.drop(self.reference_column, axis=1, inplace=True)
        except ValueError:
            pass
        return return_data

    def fit(self, *_):
        return self


# This will be awesome as soon as Patsy is pickle-able
# class patsy_formula_transformer(TransformerMixin):
#     """
#     EXPERIMENTAL - not in use
#
#     In: pd.Series

#     Out: pd.Series
#     """

#     def __init__(self, formula, design_matrix=None):
#         self.formula = formula
#         self.design_matrix = design_matrix

#     def transform(self, data):
#         if self.design_matrix is None:
#             raise AttributeError('`fit` must be called before transform to created `self.design_matrix`')
#         return dmatrix(self.design_matrix.design_info, data=data, return_type='dataframe')

#     def fit(self, data, *_):
#         if self.design_matrix is None:
#             self.design_matrix = dmatrix(formula_like=self.formula,
#                                          data=data,
#                                          return_type='matrix',
#                                          NA_action='raise')
#         return self

class model_average(RegressorMixin):
    def predict(self, data):
        data['ensemble_prediction'] = data.apply(np.mean, axis=1)
        return data

    def transform(self, data, *some_args):
        return data

    def fit(self, *args, **kwargs):
        return self
