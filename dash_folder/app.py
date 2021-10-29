# file app.py
import dash
import pandas as pd
from dash import dcc
from dash import html
import plotly.graph_objects as go
from utils.data_handling import load_data
import numpy as np
from dash.dependencies import Input, Output, State
from dash_folder.template import assets_folder, colors
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.metric_evaluation import evaluate_metric_quadratic
import pandas as pd

app = dash.Dash(__name__, assets_folder=assets_folder)

all = load_data('latent_classification')
train = pd.DataFrame(all.iloc[:80].append(all.iloc[100:180]), index=[i for i in range(160)])
test = pd.DataFrame(all.iloc[80:100].append(all.iloc[180:]), index=[i for i in range(40)])

min_x = all['x'].min()
max_x = all['x'].max()
min_y = all['y'].min()
max_y = all['y'].max()
min_z = all['z'].min()
max_z = all['z'].max()

X_START = 0.02
delta = 0.1

yy, zz = np.meshgrid(np.arange(min_y, max_y, delta), np.array([min_z, max_z]))
xx = np.ones(yy.shape)


def scatter(x, y, z, name, color):
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        name=name,
        mode='markers',
        marker=dict(
            size=5,
            color=color,
            opacity=1
        )
    )


plane = go.Surface(
    x=xx * X_START,
    y=yy,
    z=zz,
    colorscale=['grey' for i in range(len(xx))],
    opacity=0.8,
    showscale=False,
    name='Classification Plane',
    showlegend=True
)

graph_layout_default_settings = {'scene': {
    'camera': {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': -0.07214567600111127, 'y': -0.08497201560137722, 'z': -0.27943759806176177},
        'eye': {'x': -0.5135394958253185, 'y': -1.9748036183855688, 'z': 0.7264046470993168},
        'projection': {'type': 'perspective'},
    },
    'xaxis_title': 'Anger(-),Sad(+)-Dim',
    'yaxis_title': 'PCA 1',
    'zaxis_title': 'PCA 2',
}}

app.layout = html.Div(
    children=[
        html.Div([
            html.Div([
                html.Div(id='accuracy'),
                html.Div(id='precision'),
                html.Div(id='recall'),
                html.Div([
                    dcc.Dropdown(
                        id='data-type',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Training', 'value': 'train'},
                            {'label': 'Test', 'value': 'test'}
                        ],
                        value='all',
                        style={
                            'width': '100%',
                            'display': 'flex',
                            'flex-direction': 'column',
                        }
                    )
                ],
                    style={
                        'width': '80%',
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
                        style={
                            'width': '100%',
                        }
                    )
                ],
                    style={
                        'width': '80%',
                    }
                ),
            ],
                id='metrics',
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
                        'width':'100%',
                    }
                ),
                html.Div([
                    dcc.Slider(
                        id='plane',
                        min=min_x,
                        max=max_x,
                        step=delta,
                        value=X_START,
                        marks={
                            X_START: f"{X_START:.2f} (SVM)",
                            min_x: f"{min_x:.2f} (Min)",
                            max_x: f"{max_x:.2f} (Max)"
                        },
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
                'height':'100vh',
                'display':'flex',
                'flex-direction':'column',
                'align-items': 'center',
                'width':'70%',
            }),

        ], style={
            'display': 'flex',
        })

    ]
)


def copy_relayout_data(layout, relayout_data):
    # Keep zoom after updating graph
    if relayout_data:
        if 'scene.camera' in relayout_data:
            layout.scene.camera.update(relayout_data['scene.camera'])
    return layout


def get_trigger():
    ctx = dash.callback_context
    if ctx.triggered:
        return ctx.triggered[0]["prop_id"].split(".")[0]
    else:
        return None


@app.callback(Output('graph', 'figure'),
              Output('accuracy', 'children'),
              Output('precision', 'children'),
              Output('recall', 'children'),
              Output('accuracy', 'style'),
              Output('precision', 'style'),
              Output('recall', 'style'),
              Input('plane', 'value'),
              Input('classification-type', 'value'),
              Input('data-type', 'value'),
              State('graph', 'relayoutData'),
              State('graph', 'figure'))
def change_plane_x(slider_value, classification_type, data_type, relayout_data, prev_graph):
    trigger = get_trigger()

    if data_type == 'train':
        sample_df = pd.DataFrame(train)
    elif data_type == 'test':
        sample_df = pd.DataFrame(test)
    else:
        sample_df = pd.DataFrame(all)

    predicted = np.where(sample_df['x'] < slider_value, 0, 1)

    if trigger == 'plane' or not prev_graph:
        plane = go.Surface(
            x=xx * slider_value,
            y=yy,
            z=zz,
            colorscale=['grey' for i in range(len(xx))],
            opacity=0.8,
            showscale=False
        )
    else:
        plane = prev_graph['data'][-1]

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
        data = [scatter(c[i]['x'], c[i]['y'], c[i]['z'], cn[i], cc[i]) for i in range(len(c))]
        data.append(plane)
    elif classification_type == 'cor':
        correct_mask = sample_df['label'] == pd.Series(predicted)
        cs = sample_df[correct_mask]
        ws = sample_df[correct_mask == False]
        correct_scatter = scatter(cs['x'], cs['y'], cs['z'], 'Correct classification', colors.turquoise)
        wrong_scatter = scatter(ws['x'], ws['y'], ws['z'], 'Wrong classification', colors.pink)
        data = [correct_scatter, wrong_scatter, plane]
    else:
        class_mask = sample_df['label'] == True
        tr = sample_df[class_mask]
        fa = sample_df[class_mask == False]
        true_scatter = scatter(tr['x'], tr['y'], tr['z'], 'Sad', colors.blue)
        false_scatter = scatter(fa['x'], fa['y'], fa['z'], 'Anger', colors.yellow)
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

    layout = go.Layout()
    if relayout_data and 'camera' in relayout_data:
        layout = copy_relayout_data(layout, relayout_data)
    else:
        layout.update(graph_layout_default_settings)
    return go.Figure(data=data,
                     layout=layout), accuracy_label, precision_label, recall_label, accuracy_style, precision_style, recall_style


if __name__ == '__main__':
    app.run_server(debug=True)
