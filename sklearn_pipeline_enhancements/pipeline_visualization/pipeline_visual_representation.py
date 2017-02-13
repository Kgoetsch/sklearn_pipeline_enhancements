import pandas as pd


class PipelineVisualRepresentation(object):
    def __init__(self, name, type_name, length=None, width=None, x_pos=None, y_pos=None, color=None):
        self.color = color
        self.children = []
        self.name = name
        self.type_name = type_name
        self.length = length
        self.width = width
        self.x_pos = x_pos
        self.y_pos = y_pos

    def as_dataframe(self):
        return pd.DataFrame({'width': [self.width],
                             'length': [self.length],
                             'name': [self.name],
                             'type_name': [self.type_name],
                             'x_pos': [self.x_pos],
                             'y_pos': [self.y_pos],
                             'color': [self.color]})
