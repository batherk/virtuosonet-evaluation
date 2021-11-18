import dash
import plotly.graph_objects as go


def get_trigger():
    ctx = dash.callback_context
    if ctx.triggered:
        return ctx.triggered[0]["prop_id"].split(".")[0]
    else:
        return None


def scatter3d(x, y, z, name, color):
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

graph_layout_default_settings = {
    'scene': {
        'camera': {
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': {'x': -0.07214567600111127, 'y': -0.08497201560137722, 'z': -0.27943759806176177},
            'eye': {'x': -0.5135394958253185, 'y': -1.9748036183855688, 'z': 0.7264046470993168},
            'projection': {'type': 'perspective'},
        },
    },
    'legend': {
        'xanchor': 'right'
    },
    'margin': {
        'autoexpand': False
    },
}

def get_layout(relayout_data, prev_figure):
    layout = go.Layout()
    if prev_figure:
        layout.update(prev_figure['layout'])
        if relayout_data and 'scene.camera' in relayout_data:
            layout.scene.camera.update(relayout_data['scene.camera'])
    else:
        layout.update(graph_layout_default_settings)
    return layout
