import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash_folder.template import assets_folder
from utils.data_handling import load_data
from utils.dimension_manipulation import get_coordinates
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

app = dash.Dash(__name__, assets_folder=assets_folder, title='ROC')
RUN_PORT = 8053

data_df = load_data('styles')
dimension_df = load_data('disentangled_dimensions_all_combinations')
dimension_vectors = dimension_df.loc[:, 'l0':].to_numpy()

curves = []
auc_scores = []

for i in range(len(dimension_vectors)):
    negative_name = dimension_df.iloc[i]['negative_name']
    positive_name = dimension_df.iloc[i]['positive_name']
    intercept = dimension_df.iloc[i]['direction_intercept'][0]
    mask = (data_df['style_name'] == negative_name) | (data_df['style_name'] == positive_name)

    latent_vectors = data_df[mask].loc[:, 'l0':].to_numpy()
    y_score = get_coordinates(latent_vectors, dimension_vectors[i], 1).squeeze()

    style_name = data_df[mask]['style_name'].reset_index(drop=True)
    y_true = style_name.apply(lambda x: 1 if x == positive_name else 0).to_numpy()

    xs, ys, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    auc_scores.append(auc_score)
    curves.append(go.Scatter(
        x=xs,
        y=ys,
        name=f"{negative_name} - {positive_name}: {auc_score:.2f}",
    ))

auc_scores = np.array(auc_scores)

app.layout = html.Div(children=[
    html.Div([
        f"Average AUC score: {auc_scores.mean():.2f}"
    ], style={
        'width':'100vw',
        'display': 'flex',
        'justify-content':'center',


    }),
    dcc.Graph(
        figure=go.Figure(data=curves),
        style=dict(
            height='80vh',
            width='60vw'
        )
    ),

], style={
    'display': 'flex',
    'flex-direction': 'column',
    'justify-content':'space-evenly',
    'align-items':'center',
    'height': '100vh',
    'width': '100vw',
})

if __name__ == '__main__':
    app.run_server(debug=True, port=RUN_PORT)
