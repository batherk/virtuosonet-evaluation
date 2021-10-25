import torch
import os
from utils.paths import get_root_folder

FOLDER_PATH = os.path.join(get_root_folder(), 'virtuosoNet', '')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "isgn"
HIER_TYPE = "han_ar_measure"
LOSS_TYPE = "MSE"
HIERARCHY = False
HIERARCHY_TEST = False
IN_HIER = False
HIER_MEAS = False
HIER_BEAT = False
HIER_MODEL = None
RAND_TRAIN = True
TRILL = False
DISKLAVIER = False
PEDAL = False
RESUME_TRAINING = False
START_TEMPO = 0
VELOCITY = '50,65'

learning_rate = 0.0003
TIME_STEPS = 500
VALID_STEPS = 5000
DELTA_WEIGHT = 2
NUM_UPDATED = 0
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5
KLD_MAX = 0.01
KLD_SIG = 20e4
NUM_EPOCHS = 100
NUM_KEY_AUGMENTATION = 1

NUM_INPUT = 78
NUM_OUTPUT = 11
NUM_PRIME_PARAM = 11
NUM_TEMPO_PARAM = 1
VEL_PARAM_IDX = 1
DEV_PARAM_IDX = 2
PEDAL_PARAM_IDX = 3
NUM_SECOND_PARAM = 0
NUM_TRILL_PARAM = 5
NUM_VOICE_FEED_PARAM = 0  # Velocity, onset deviation
NUM_TEMPO_INFO = 0
NUM_DYNAMIC_INFO = 0  # Distance from marking, dynamics vector 4, mean_piano, forte marking and velocity = 4
IS_TRILL_INDEX_SCORE = -11
IS_TRILL_INDEX_CONCATED = -11 - (NUM_PRIME_PARAM + NUM_SECOND_PARAM)

SLUR_EDGE = False
VOICE_EDGE = True
QPM_INDEX = 0
TEMPO_IDX = 26
QPM_PRIMO_IDX = 4
TEMPO_PRIMO_IDX = -2
GRAPH_KEYS = ["onset", "forward", "melisma", "rest"]
N_EDGE_TYPE = 10
BATCH_SIZE = 1


def update_settings(
    model_type=MODEL_TYPE,
    hier_type=HIER_TYPE,
    slur_edge=SLUR_EDGE,
    voice_edge=VOICE_EDGE,
    num_input=NUM_INPUT,
    num_prime_param=NUM_PRIME_PARAM,
):

    global hier_meas
    global hier_beat
    if "measure" in model_type or "beat" in model_type:
        hierarchy = True
    elif "note" in model_type:
        in_hier = True  # In hierarchy mode
    if hierarchy or in_hier:
        if "measure" in model_type or "measure" in hier_type:
            hier_meas = True
        elif "beat" in model_type or "beat" in hier_type:
            hier_beat = True

    trill = False
    if "trill" in model_type:
        trill = True

    graph_keys = ["onset", "forward", "melisma", "rest"]

    if slur_edge:
        graph_keys.append("slur")
    if voice_edge:
        graph_keys.append("voice")

    n_edge_type = len(graph_keys) * 2

    if hierarchy:
        num_output = 2
    elif trill:
        num_input += num_prime_param
        num_output = 5
    else:
        num_output = 11
    if in_hier:
        num_input += 2
    return (
        model_type,
        hier_type,
        hierarchy,
        in_hier,
        hier_type,
        hier_meas,
        hier_beat,
        trill,
        graph_keys,
        slur_edge,
        voice_edge,
        n_edge_type,
        num_input,
        num_output,
        num_prime_param,
    )
