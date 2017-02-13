from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm.base import BaseSVC

from pipeline_visual_representation import PipelineVisualRepresentation
from sklearn_pipeline_enhancements.shared.transformers import DataFrameUnion

buffer_size = 1


def parse_sklearn_pipeline(puzzle_piece, x_position=0, y_position=0):
    representation = None
    if isinstance(puzzle_piece, tuple):
        puzzle_piece_name = puzzle_piece[0]
        puzzle_piece_thing = puzzle_piece[1]

        # DataFrameUnion
        if isinstance(puzzle_piece_thing, DataFrameUnion) or isinstance(puzzle_piece_thing, FeatureUnion):
            representation = parse_DataFrameUnion(puzzle_piece_name, puzzle_piece_thing, x_position, y_position)
            x_position += representation.width + buffer_size / 2

        # Pipeline
        elif isinstance(puzzle_piece_thing, Pipeline):
            representation = parse_pipeline(puzzle_piece_name, puzzle_piece_thing, x_position, y_position)
            y_position += representation.length + buffer_size / 2

        elif isinstance(puzzle_piece_thing, TransformerMixin) or \
                isinstance(puzzle_piece_thing, BaseSVC) or \
                isinstance(puzzle_piece_thing, RegressorMixin) or \
                isinstance(puzzle_piece_thing, BaseEstimator):
            representation = parse_transformer(puzzle_piece_name, puzzle_piece_thing, x_position, y_position)
        else:
            print '\nMISSING THING'
            print puzzle_piece_name
            raise
    else:
        print 'WARNING: DID NOT PASS A TUPLE'

    return representation


def parse_transformer(puzzle_piece_name, puzzle_piece_thing, x_position, y_position):
    transformer_representation = PipelineVisualRepresentation(name=puzzle_piece_name, type_name='Transformer', length=1,
                                                              width=1, x_pos=x_position, y_pos=y_position,
                                                              color='salmon')
    return transformer_representation


def parse_pipeline(puzzle_piece_name, puzzle_piece_thing, x_position, y_position):
    pipe_representation = PipelineVisualRepresentation(name=puzzle_piece_name, type_name='sklean.Pipeline',
                                                       x_pos=x_position,
                                                       y_pos=y_position, color='dodgerblue')
    if puzzle_piece_name == 'pipeline-1':
        pass
    parent_width = 0
    parent_length = 0  # buffer_size
    local_x_position = buffer_size / 2
    local_y_position = buffer_size / 2
    for base_estimator in puzzle_piece_thing.steps:
        representation = parse_sklearn_pipeline(base_estimator, local_x_position, local_y_position)
        pipe_representation.children.append(representation)

        local_y_position += representation.length + buffer_size
        parent_width = max(parent_width, representation.width + buffer_size)
        parent_length += representation.length + buffer_size

    pipe_representation.length = parent_length
    pipe_representation.width = parent_width
    return pipe_representation


def parse_DataFrameUnion(puzzle_piece_name, puzzle_piece_thing, x_position, y_position):
    dataframe_union_representation = PipelineVisualRepresentation(name=puzzle_piece_name, type_name='DataFrameUnion',
                                                                  x_pos=x_position, y_pos=y_position,
                                                                  color='mediumspringgreen')
    parent_length = 0
    local_x_position = buffer_size / 2
    local_y_position = buffer_size / 2
    parent_width = 0  # buffer_size
    for base_estimator in puzzle_piece_thing.get_params()['transformer_list']:
        representation = parse_sklearn_pipeline(base_estimator, local_x_position, local_y_position)
        dataframe_union_representation.children.append(representation)

        local_x_position += representation.width + buffer_size
        parent_length = max(parent_length, representation.length + buffer_size)
        parent_width += representation.width + buffer_size

    dataframe_union_representation.length = parent_length
    dataframe_union_representation.width = parent_width

    return dataframe_union_representation


def parse_built_representation(representation_object, x_base=0, y_base=0):
    df = representation_object.as_dataframe()

    df['x_position'] = adjusted_x = df['x_pos'] + x_base
    df['y_position'] = adjusted_y = df['y_pos'] + y_base

    if len(representation_object.children) > 0:
        for base_estimator in representation_object.children:
            new_df = parse_built_representation(base_estimator,
                                                adjusted_x,
                                                adjusted_y)
            df = df.append(new_df, ignore_index=True).copy()

    return df


def build_pipeline_visualization(outer_most_df):
    plt.xkcd()
    max_y = max(outer_most_df['length']) + 1
    max_x = max(outer_most_df['width']) + 1

    fig1 = plt.figure(num=1, figsize=(20, 10))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_ylim(-1, max_y)
    ax1.set_xlim(-1, max_x)
    ax1.patch.set_facecolor('white')

    for index, row in outer_most_df.iterrows():
        ax1.add_patch(
            patches.Rectangle(
                (row['x_position'], row['y_position']),  # (x,y)
                row['width'],  # width
                row['length'],  # height
                #             fill=False,
                edgecolor='black',
                lw=3,
                facecolor=row['color']
            )
        )
        ax1.text(x=row['x_position'],
                 y=row['y_position'] + row['length'],
                 s=row['name'],
                 fontsize=11
                 #             horizontalalignment='right',
                 #             verticalalignment='top'
                 #             transform=ax1.transAxes
                 )
    ax1.set_axis_off()
    return ax1


def draw_pipeline_visualization(model_path):
    if isinstance(model_path, str):
        loaded_model = joblib.load(model_path)
    else:
        loaded_model = model_path

    outer_most_representation = parse_sklearn_pipeline(('loaded_model', loaded_model))
    outer_most_df = parse_built_representation(outer_most_representation,
                                               outer_most_representation.x_pos,
                                               outer_most_representation.y_pos
                                               )
    ax1 = build_pipeline_visualization(outer_most_df)
    return ax1
