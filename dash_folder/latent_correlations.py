import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash_folder.template import assets_folder, colors
from utils.data_handling import load_data


app = dash.Dash(__name__, assets_folder=assets_folder, title='Latent Correlations')
RUN_PORT = 8052

data_df = load_data('all_styles_100')
dimension_df = load_data('disentangled_dimensions_all_combinations')
dimension_vectors = dimension_df.loc[:, 'l0':].to_numpy()
latent_vectors = data_df.loc[:, 'l0':].to_numpy()



corr = data_df.loc[:, 'l0':].corr()

app.layout = html.Div(children=[
    html.Div([
        f"Latent Correlations"
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
                colorscale=[colors.grey1, colors.turquoise],
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
    app.run_server(debug=True, port=RUN_PORT)
