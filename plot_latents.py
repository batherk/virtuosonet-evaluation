from utils.data_handling import load_styles_df
from utils.plot import plot_latents

styles_df = load_styles_df(path='./data/', name='multiple_styles')
plot_latents(styles_df)