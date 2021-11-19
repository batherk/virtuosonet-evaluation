import dash
import pandas as pd
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash_folder.template import assets_folder, colors
from utils.data_handling import load_data
from utils.dimension_manipulation import get_coordinates

app = dash.Dash(__name__, assets_folder=assets_folder, title='Dimension Correlations')
RUN_PORT = 8051

data_df = load_data('styles')
dimension_df = load_data('disentangled_dimensions')
dimension_vectors = dimension_df.loc[:, 'l0':].to_numpy()
latent_vectors = data_df.loc[:, 'l0':].to_numpy()


def get_axis_name(df, index, show_direction=True):
    return f"{df.iloc[index]['negative_name']} {'(-)' if show_direction else ''} " \
           f"- {df.iloc[index]['positive_name']} {'(+)' if show_direction else ''}"


features = get_coordinates(latent_vectors, dimension_vectors)
features_df = pd.DataFrame(features,
                           columns=[get_axis_name(dimension_df, i, False) for i in range(len(dimension_df.index))])

corr = features_df.corr()

app.layout = html.Div(children=[
    html.Div([
        f"Dimension correlations"
    ], style={
        'width': '100vw',
        'display': 'flex',
        'justify-content': 'center',

    }),
    dcc.Graph(
        figure=go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.index.values,
                y=corr.columns.values,
                colorscale=[colors.yellow,colors.grey1, colors.turquoise],
            ),
        ),
        style=dict(
            height='80vh',
            width='60vw'
        )
    ),

], style={
    'display': 'flex',
    'flex-direction': 'column',
    'justify-content': 'center',
    'align-items': 'center',
    'height': '100vh',
    'width': '100vw',
})

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
