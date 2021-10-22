from utils.data_generation import generate_styles_df, generate_multiple_styles_df
from utils.data_handling import save_data

styles_df = generate_styles_df()
multiple_styles_df = generate_multiple_styles_df()

save_data(styles_df, 'styles')
save_data(multiple_styles_df, 'multiple_styles')