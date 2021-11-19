import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash_folder.template import assets_folder, colors
from utils.data_handling import load_data
from utils.dash import get_trigger, scatter3d, get_layout
from utils.dimension_manipulation import get_pca_coords
import plotly.graph_objects as go

app = dash.Dash(__name__, assets_folder=assets_folder, title='Latent Correlations')
RUN_PORT = 8054
DIMENSIONS = 3

data_df = load_data('styles')
base_option = {'label': 'All', 'value': 'All'}
composer_options = [base_option]
composer_options += [{'label': composer, 'value': composer} for composer in data_df['composer'].unique()]
average_options = [{'label': '10 per Performance', 'value': 'All'},
                   {'label': 'Average per Performance', 'value': 'Average'}]

app.layout = html.Div([
    html.Div([
        f"Clustering Based on Emotion Dataset"
    ], style={
        'width': '100vw',
        'display': 'flex',
        'justify-content': 'center',

    }), html.Div([
        html.Div([
            dcc.Dropdown(
                id='composer',
                options=composer_options,
                value=composer_options[0]['value'],
                clearable=False,
                style={'width': '100%'},
            ),
            dcc.Dropdown(
                id='piece',
                options=[base_option],
                value=base_option['value'],
                clearable=False,
                style={'width': '100%'},
            ), dcc.Dropdown(
                id='average',
                options=average_options,
                value=average_options[0]['value'],
                clearable=False,
                style={'width': '100%'},
            ), ],

            style={
                'display': 'flex',
                'flex-direction': 'column',
                'height': '100%',
                'width': '30vw',
                'align-items': 'center',
                'justify-content': 'space-evenly'

            }
        ),
        dcc.Graph(
            id='clustering',
            style=dict(
                height='80vh',
                width='60vw'
            )),
    ],
        style={
            'display': 'flex',
            'justify-content': 'space-evenly',
        }
    ),

], style={
    'display': 'flex',
    'flex-direction': 'column',
    'justify-content': 'center',
    'align-items': 'center',
    'height': '100vh',
    'width': '100vw',
})


@app.callback(Output('clustering', 'figure'),
              Output('piece', 'value'),
              Output('piece', 'options'),
              Input('composer', 'value'),
              Input('piece', 'value'),
              Input('average', 'value'),
              State('piece', 'options'),
              State('clustering', 'figure'),
              State('clustering', 'relayoutData'))
def change_piece_options(composer, piece, average, piece_options, prev_figure, relayout_data):
    trigger = get_trigger()

    if trigger == 'composer':
        if composer == base_option['value']:
            piece_options, piece = [base_option], base_option['value']
        else:
            composer_mask = data_df['composer'] == composer
            pieces = data_df[composer_mask]['piece'].unique()
            options = [base_option] + [{'label': piece, 'value': piece} for piece in pieces]
            value = base_option['value']
            piece_options, piece = options, value

    filtered = data_df.copy()

    if composer != base_option['value']:
        filtered = filtered[filtered['composer'] == composer]
    if piece != base_option['value']:
        filtered = filtered[filtered['piece'] == piece]
    if average == 'Average':
        filtered = filtered.groupby(['piece', 'style_name']).mean().reset_index()

    pca_coords = get_pca_coords(filtered.loc[:, 'l0':].to_numpy(), DIMENSIONS).transpose()

    for dim in range(DIMENSIONS):
        filtered[f"PCA{dim}"] = pca_coords[dim]

    emotions = filtered['style_name'].unique()

    scatterplots = []

    for i, emotion in enumerate(emotions):
        emotion_filtered = filtered[filtered['style_name'] == emotion]
        scatterplots.append(scatter3d(
            x=emotion_filtered['PCA0'],
            y=emotion_filtered['PCA1'],
            z=emotion_filtered['PCA2'],
            name=emotion,
            color=None
        ))

    layout = get_layout(relayout_data, prev_figure)
    figure = go.Figure(data=scatterplots, layout=layout)

    return figure, piece, piece_options


if __name__ == '__main__':
    app.run_server(debug=True, port=RUN_PORT)
