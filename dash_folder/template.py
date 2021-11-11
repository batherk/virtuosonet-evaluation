import os

import plotly.graph_objs as go
import plotly.io as pio

DARK_MODE = True


class SolutionSeekerColors:
    black = 'rgb(0,0,2)'
    grey1 = 'rgb(34,35,42)'
    grey2 = 'rgb(61,62,63)'
    grey3 = 'rgb(120,120,120)'
    grey4 = 'rgb(198,198,198)'
    white = 'rgb(255,255,255)'
    yellow = 'rgb(255,227,29)'
    byellow = 'rgb(255,244,166)'
    turquoise = 'rgb(166,255,198)'
    bturquoise = 'rgb(220,255,232)'
    blue = 'rgb(108,172,252)'
    blue_dark = 'rgb(77,105,217)'
    pink = 'rgb(223,154,156)'
    purple = 'rgb(171,75,139)'


colors = SolutionSeekerColors()
layout = go.layout.Template()

if DARK_MODE:
    layout.layout.plot_bgcolor = colors.grey1
    layout.layout.paper_bgcolor = colors.grey1
    layout.layout.font = {'color': colors.white}
    layout.layout.xaxis = {'gridcolor': colors.grey2}
    layout.layout.yaxis = {'gridcolor': colors.grey2}
    layout.layout.colorway = [colors.yellow,
                              colors.turquoise,
                              colors.blue,
                              colors.byellow,
                              colors.bturquoise]
else:
    layout.layout.colorway = [colors.yellow,
                              colors.turquoise,
                              colors.blue,
                              colors.pink,
                              colors.purple,
                              colors.blue_dark
                              ]
assets_folder = os.path.dirname(os.path.realpath(__file__)) + "/assets"
pio.templates['seeker'] = layout
pio.templates.default = 'seeker'
