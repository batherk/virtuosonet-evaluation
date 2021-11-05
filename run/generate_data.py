from utils.data_generation import generate_multiple_styles_df
from utils.data_handling import save_data

styles_df = generate_multiple_styles_df(amount=100)
save_data(styles_df, name='all_styles_100')