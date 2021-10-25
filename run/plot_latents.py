from utils.data_handling import load_data
from utils.plot import plot_latents

styles_df = load_data(name='multiple_styles')
plot_latents(styles_df)