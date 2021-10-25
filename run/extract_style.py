#%% Imports

from utils import virtuoso_running as vh

#%% Settings

MODEL_TYPE = "isgn"

PATH = "test_pieces/emotionNet/Bach_Prelude_1/"
PERFORMANCE_NAME = "Relax_sub1"
COMPOSER_NAME = "Bach"
DATA_FILE = "training_data"

#%% Load Model

model = vh.load_model(MODEL_TYPE)

#%% Load means and stds from data file

means, stds, _, _, _, _, _ = vh.load_stat_file(DATA_FILE)

#%% Encode Song

z_vector, qpm_primo = vh.load_file_and_encode_style(
    PATH, PERFORMANCE_NAME, COMPOSER_NAME, model, means, stds
)

#%%

print(qpm_primo, z_vector)