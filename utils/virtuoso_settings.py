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
initialize_model = False

# Score
score_path = "emotionNet/Bach_Prelude_1/musicxml_cleaned.musicxml"
composer = "Bach"

# Div
lr = 1e-4
weight_decay = 1e-5
delta_weight = 1
world_size = 1
velocity_multiplier = 1
clock_interval_in_16th = 4
len_valid_slice = 10000
num_workers = 0
meas_loss_weight = 1
lr_decay_step = 5000
lr_decay_rate = 0.98

qpm_primo = None
is_hier = False
hier_meas = False
hier_beat = False
meas_note = False
is_trill = False
intermediate_loss = True
tempo_loss_in_note = False
vel_balance_loss = False
delta_loss = False
make_log = True
pin_memory = True
save_cluster = False
multi_instruments = False
tempo_clock = False
boolPedal = False
disklavier = True
save_csv = False
save_cluster = False
mod_midi_path = None
resume_training = False

# Paths
saved_folder = os.path.join(get_root_folder(), "saved")

virtuosonet_folder_path = os.path.join(get_root_folder(), "virtuosonet", "")
yml_path = os.path.join(saved_folder, "models", f"{model_type}.yml")
checkpoint = os.path.join(saved_folder, "models", f"{model_type}_{checkpoint_name}.pt")
xml_path = Path(os.path.join(virtuosonet_folder_path + "test_pieces", score_path))
emotion_data_path = Path(os.path.join(get_root_folder(), "data", "emotion"))
output_path = results_folder
checkpoints_dir = Path(os.path.join(saved_folder, "checkpoints"))
logs = Path(os.path.join(saved_folder, "logs"))


# Settings based on previous settings
config = read_model_setting(yml_path)
net_param = config.nn_params
graph_keys = net_param.graph_keys
meas_note = net_param.meas_note

