#%%

from virtuosoNet import functions as vif
from Style import relax, sad


#%% Settings

MODEL_TYPE = "isgn"
TRILL_MODEL = 'trill_default'
PATH = "test_pieces/emotionNet/Bach_Prelude_1/"
COMPOSER_NAME = "Bach"
DATA_FILE = "training_data"

#%% Defining style
mix = relax.approach(sad, 0.5, 'Sad_Relax_Mix')
overcorrection = relax.approach(sad, 1.2, 'Overcorrection')

latent = mix.to_dict()

#%% Load Model

model = vif.load_model(MODEL_TYPE)
trill_model = vif.load_model(TRILL_MODEL)

#%% Load means and stds from data file

means, stds, bins, _, _, _, _ = vif.load_stat_file(DATA_FILE)

#%% Stylize Song

vif.load_file_and_generate_performance(PATH, COMPOSER_NAME, latent, means, stds, bins, model, MODEL_TYPE, trill_model)