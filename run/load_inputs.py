#%% Imports

import utils.virtuoso_running as vh
import virtuosoNet.pyScoreParser.xml_matching as xml_matching
from utils.virtuoso_settings import FOLDER_PATH
from utils.virtuoso_handling import get_input_df

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


absolute_path = FOLDER_PATH + PATH


test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(
        absolute_path, PERFORMANCE_NAME, COMPOSER_NAME, means, stds
    )

inputs = get_input_df(test_x, test_y)

print(inputs)