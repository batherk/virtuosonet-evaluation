import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from utils.data_handling import load_data
import numpy as np
from dash.dependencies import Input, Output, State
from dash_folder.template import assets_folder, colors
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.metric_evaluation import evaluate_metric_quadratic
from utils.dimension_manipulation import get_coordinates
import pandas as pd
from utils.dash import get_layout, get_trigger, scatter3d

app = dash.Dash(__name__, assets_folder=assets_folder, title='Classification')
RUN_PORT = 8050

dimensions = 3
SLIDER_STEPS = 50
PCA_DIMENSIONS = 2


data_df = load_data('styles')
dimension_df = load_data('disentangled_dimensions')
dimension_vectors = dimension_df.loc[:, 'l0':].to_numpy()


def get_axis_name(df, index, show_direction=True):
    return f"{df.iloc[index]['negative_name']} {'(-)' if show_direction else ''} " \
           f"- {df.iloc[index]['positive_name']} {'(+)' if show_direction else ''}"


axis_1_options = [
    {'label': get_axis_name(dimension_df, i, show_direction=False), 'value': f"dim{i + 1}"} for i in
    range(len(dimension_df.index))
]

other_axis_options = axis_1_options + [
    {'label': f"PCA {i + 1}", 'value': f"pca{i + 1}"} for i in
    range(PCA_DIMENSIONS)
]


app.layout = html.Div(
    children=[
        html.Div([
            html.Div([
                html.Div([
                    html.Div(id='accuracy'),
                    html.Div(id='precision'),
                    html.Div(id='recall'),
                ],
                    id='metrics',
                    style={
                        'display': 'flex',
                        'flex-direction': 'column',
                        'justify-content': 'space-evenly',
                        'height': '50%',
                    }
                ),
                html.Div([
                    dcc.Dropdown(
                        id='classification-type',
                        options=[
                            {'label': 'Classes', 'value': 'classes'},
                            {'label': 'Correct Classifications', 'value': 'cor'},
                            {'label': 'True/False Negatives/Positives', 'value': 'tfnp'}
                        ],
                        value='classes',
                        clearable=False,
                        style={
                            'width': '100%',
                        }
                    )
                ],
                    id='dropdowns',
                    style={
                        'width': '80%',
                        'display': 'flex',
                        'flex-direction': 'column',
                        'justify-content': 'space-evenly',
                        'height': '30%'
                    }
                ),
                html.Div([
                    dcc.Dropdown(
                        id='axis-1',
                        options=axis_1_options,
                        value='dim1',
                        clearable=False,
                        style={
                            'width': '100%',
                        }
                    ),
                    dcc.Dropdown(
                        id='axis-2',
                        options=other_axis_options,
                        value='dim2',
                        clearable=False,
                        style={
                            'width': '100%',
                        }),
                    dcc.Dropdown(
                        id='axis-3',
                        options=other_axis_options,
                        value='dim3',
                        clearable=False,
                        style={
                            'width': '100%',
                        }
                    )
                ],
                    id='axes',
                    style={
                        'width': '80%',
                        'display': 'flex',
                        'flex-direction': 'column',
                        'justify-content': 'space-evenly',
                        'height': '30%'
                    }
                ),
            ],
                id='sidebar-left',
                style={
                    'display': 'flex',
                    'flex-direction': 'column',
                    'justify-content': 'space-evenly',
                    'align-items': 'center',
                    'width': '30vw',
                }
            ),
            html.Div([
                dcc.Graph(
                    id='graph',
                    style={
                        'height': '90vh',
                        'width': '100%',
                    }
                ),
                html.Div([
                    dcc.Slider(
                        id='plane',
                        tooltip={
                            'always_visible': False,
                            'placement': 'top'
                        },
                        updatemode='drag'
                    )
                ], style={
                    'height': '10vh',
                    'align-items': 'center',
                    'justify-content': 'center',
                    'width': '50vw',
                })
            ],
                style={
                    'height': '100vh',
                    'display': 'flex',
                    'flex-direction': 'column',
                    'align-items': 'center',
                    'width': '70%',
                }),

        ], style={
            'display': 'flex',
        })

    ]
)



@app.callback(Output('plane', 'marks'),
              Output('plane', 'min'),
              Output('plane', 'max'),
              Output('plane', 'value'),
              Output('plane', 'step'),
              Input('axis-1', 'value'))
def change_classification_axis(axis_1):
    dim = int(axis_1[-1]) - 1
    negative_name = dimension_df.iloc[dim]['negative_name']
    positive_name = dimension_df.iloc[dim]['positive_name']
    mask = (data_df['style_name'] == negative_name) | (data_df['style_name'] == positive_name)

    latent_vectors = data_df[mask].loc[:, 'l0':].to_numpy()
    coordinates = get_coordinates(latent_vectors, dimension_vectors)
    values = coordinates.transpose()[dim]
    intercept = dimension_df.iloc[dim]['direction_intercept'][0]
    marks = {
        values.min(): f"Min: {values.min():.2f}",
        values.max(): f"Max: {values.max():.2f}",
        intercept: f"SVM: {intercept:.2f}"
    }
    steps = (values.max()-values.min())/SLIDER_STEPS
    return marks, values.min(), values.max(), intercept, steps


@app.callback(Output('graph', 'figure'),
              Output('accuracy', 'children'),
              Output('precision', 'children'),
              Output('recall', 'children'),
              Output('accuracy', 'style'),
              Output('precision', 'style'),
              Output('recall', 'style'),
              Input('plane', 'value'),
              Input('classification-type', 'value'),
              Input('axis-1', 'value'),
              Input('axis-2', 'value'),
              Input('axis-3', 'value'),
              State('graph', 'figure'),
              State('graph', 'relayoutData'))
def change_plane_x(slider_value, classification_type, axis_1, axis_2, axis_3, prev_figure, relayout_data):
    trigger = get_trigger()

    layout = get_layout(relayout_data, prev_figure)

    dim = int(axis_1[-1]) - 1
    negative_name = dimension_df.iloc[dim]['negative_name']
    positive_name = dimension_df.iloc[dim]['positive_name']
    mask = (data_df['style_name'] == negative_name) | (data_df['style_name'] == positive_name)
    intercept = dimension_df.iloc[dim]['direction_intercept'][0]

    latent_vectors = data_df[mask].loc[:, 'l0':].to_numpy()
    coordinates = get_coordinates(latent_vectors, dimension_vectors, len(dimension_vectors) + PCA_DIMENSIONS)

    style_name = data_df[mask]['style_name'].reset_index(drop=True)
    labels = style_name.apply(lambda x: 1 if x == positive_name else 0).rename('label')

    all = pd.concat([labels, pd.DataFrame(coordinates, columns=[dic['value'] for dic in other_axis_options])], axis=1)

    columns = []
    column_names = ['x', 'y', 'z']
    sample_df = pd.DataFrame(labels)
    axis_names = []

    for i, axis in enumerate([axis_1, axis_2, axis_3]):
        dim = int(axis[-1]) - 1
        if 'pca' in axis:
            axis_names.append(f"PCA {dim + 1}")
        else:
            axis_names.append(get_axis_name(dimension_df, dim))
        sample_df[column_names[i]] = all[axis]



    layout.update({'scene': {
        'xaxis_title': axis_names[0],
        'yaxis_title': axis_names[1],
        'zaxis_title': axis_names[2],
    }})

    predicted = pd.Series(np.where(sample_df['x'] < slider_value, 0, 1), index=sample_df.index)

    if not prev_figure:
        yy, zz = np.meshgrid(np.arange(sample_df['y'].min(), sample_df['y'].max(),
                                       (sample_df['x'].max() - sample_df['x'].min()) / SLIDER_STEPS),
                             np.array([sample_df['z'].min(), sample_df['z'].max()]))
        xx = np.ones(yy.shape)
        plane = go.Surface(
            x=xx * intercept,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False,
            name='Classification Plane',
            showlegend=True
        )

    elif trigger == 'plane':
        yy, zz = np.meshgrid(np.arange(sample_df['y'].min(), sample_df['y'].max(),
                                       (sample_df['x'].max() - sample_df['x'].min()) / SLIDER_STEPS),
                             np.array([sample_df['z'].min(), sample_df['z'].max()]))
        xx = np.ones(yy.shape)
        plane = go.Surface(
            x=xx * slider_value,
            y=yy,
            z=zz,
            colorscale=['grey' for _ in range(len(xx))],
            opacity=0.8,
            showscale=False,
            name='Classification Plane',
            showlegend=True
        )
    else:
        plane = prev_figure['data'][-1]

    if classification_type == 'tfnp':
        correct = sample_df['label'] == pd.Series(predicted)
        positive = sample_df['label'] == True
        fp = sample_df[(correct == False) & (positive)]
        fn = sample_df[(correct == False) & (positive == False)]
        tp = sample_df[(correct) & (positive)]
        tn = sample_df[(correct) & (positive == False)]
        c = [tp, tn, fp, fn]
        cc = [colors.turquoise, colors.blue, colors.pink, colors.purple]
        cn = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
        data = [scatter3d(c[i]['x'], c[i]['y'], c[i]['z'], cn[i], cc[i]) for i in range(len(c))]
        data.append(plane)

    elif classification_type == 'cor':
        correct_mask = sample_df['label'] == pd.Series(predicted)
        cs = sample_df[correct_mask]
        ws = sample_df[correct_mask == False]
        correct_scatter = scatter3d(cs['x'], cs['y'], cs['z'], 'Correct classification', colors.turquoise)
        wrong_scatter = scatter3d(ws['x'], ws['y'], ws['z'], 'Wrong classification', colors.pink)
        data = [correct_scatter, wrong_scatter, plane]

    else:
        class_mask = sample_df['label'] == True
        tr = sample_df[class_mask]
        fa = sample_df[class_mask == False]
        true_scatter = scatter3d(tr['x'], tr['y'], tr['z'], positive_name, colors.blue)
        false_scatter = scatter3d(fa['x'], fa['y'], fa['z'], negative_name, colors.yellow)
        data = [true_scatter, false_scatter, plane]

    accuracy = accuracy_score(sample_df['label'], predicted)
    precision = precision_score(sample_df['label'], predicted)
    recall = recall_score(sample_df['label'], predicted)
    accuracy_label = f"Accuracy: {accuracy:.2f}"
    precision_label = f"Precision: {precision:.2f}"
    recall_label = f"Recall: {recall:.2f}"
    accuracy_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(accuracy)})"}
    precision_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(precision)})"}
    recall_style = {'border-color': f"rgb(166,255,198,{evaluate_metric_quadratic(recall)})"}

    return go.Figure(data=data,
                     layout=layout), accuracy_label, precision_label, recall_label, accuracy_style, precision_style, recall_style


if __name__ == '__main__':
    app.run_server(debug=True, port=RUN_PORT)
