import pandas as pd
from virtuosoNet import functions as vif
from utils.conversion import convert_model_z_to_latent

def generate_styles_df(
        model_type="isgn",
        path="test_pieces/emotionNet/Bach_Prelude_1/",
        style_names=['Anger', 'Relax', 'Sad', 'Enjoy', 'OR'],
        composer_name="Bach",
        data_file="training_data"
):
    model = vif.load_model(model_type)
    means, stds, _, _, _, _, _ = vif.load_stat_file(data_file)
    data = []
    latent_size = 0
    for style_name in style_names:
        performance_name = f"{style_name}_sub1"
        z_vector, qpm_primo = vif.load_file_and_encode_style(
            path, performance_name, composer_name, model, means, stds
        )
        latent = convert_model_z_to_latent(z_vector)
        latent_size = len(latent)

        data.append([style_name, qpm_primo] + latent)
    column_names = ["style_name", "qpm"] + [f"l{i}" for i in range(latent_size)]
    return pd.DataFrame(data, columns=column_names)
