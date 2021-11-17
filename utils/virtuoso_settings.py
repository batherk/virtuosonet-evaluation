import torch
import os
from utils.paths import get_root_folder, Path, results_folder
from virtuosoNet.virtuoso.utils import read_model_setting

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_code = "isgn"
model_type = "han_measnote"
checkpoint_name = "last"

# Score
score_path = "bps_5_1/musicxml_cleaned.musicxml"
composer = "Beethoven"

# Div
world_size = 1
velocity_multiplier = 1
clock_interval_in_16th = 4
len_valid_slice = 10000
num_workers = 0
qpm_primo = None

pin_memory = True
save_cluster = False
multi_instruments = False
tempo_clock = False
boolPedal = False
disklavier = True
save_csv = False
save_cluster = False
mod_midi_path = None

# Paths

virtuosonet_folder_path = os.path.join(get_root_folder(), "virtuosonet", "")
yml_path = os.path.join(get_root_folder(), "saved_models", f"{model_type}.yml")
checkpoint = os.path.join(get_root_folder(), "saved_models", f"{model_type}_{checkpoint_name}.pt")
xml_path = Path(os.path.join(virtuosonet_folder_path + "test_pieces", score_path))
emotion_data_path = Path(os.path.join(get_root_folder(), "data", "emotion"))
output_path = results_folder


# Settings based on previous settings
config = read_model_setting(yml_path)
net_param = config.nn_params
graph_keys = net_param.graph_keys
meas_note = net_param.meas_note

