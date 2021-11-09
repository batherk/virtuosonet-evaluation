import torch
import os
from utils.paths import get_root_folder, Path

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
qpm_primo = None
save_cluster = False
velocity_multiplier = 1
multi_instruments = False
tempo_clock = False
boolPedal = False
disklavier = True
clock_interval_in_16th = 4
save_csv = False
save_cluster = False
mod_midi_path = None

# Paths

virtuosonet_folder_path = os.path.join(get_root_folder(), "virtuosonet", "")
yml_path = os.path.join(get_root_folder(), "saved_models", f"{model_type}.yml")
checkpoint = os.path.join(get_root_folder(), "saved_models", f"{model_type}_{checkpoint_name}.pt")
xml_path = Path(os.path.join(virtuosonet_folder_path + "test_pieces", score_path))
output_path = Path(os.path.join(get_root_folder(), "results", ""))

