import pandas as pd


FIRST_TEST_X_LABELS = ['midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo', 'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo', 'beat_position', 'xml_position', 'grace_order','preceded_by_grace_note','followed_by_fermata_rest']
PITCH_LIST_LABELS = [f"pitch-{i}" for i in range(13)]
TEMPO_LIST_LABELS = [f"tempo-{i}" for i in range(5)]
DYNAMIC_LIST_LABELS = [f"dynamic-{i}" for i in range(5)]
TIME_SIG_VEC_LIST_LABELS = [f"tsv-{i}" for i in range(9)]
SLUR_BEAM_VEC_LIST_LABELS = [f"sbv-{i}" for i in range(6)]
COMPOSER_LIST_LABELS = [f"composer-{i}" for i in range(16)]
NOTATION_LIST_LABELS = [f"notation-{i}" for i in range(9)]
TEMPO_PRIMO_LIST_LABELS = [f"tempo-primo-{i}" for i in range(2)]

TEST_X_LABELS = FIRST_TEST_X_LABELS + PITCH_LIST_LABELS + TEMPO_LIST_LABELS + DYNAMIC_LIST_LABELS \
                + TIME_SIG_VEC_LIST_LABELS + SLUR_BEAM_VEC_LIST_LABELS + COMPOSER_LIST_LABELS \
                + NOTATION_LIST_LABELS + TEMPO_PRIMO_LIST_LABELS

FIRST_TEST_Y_LABELS = ['qpm', 'velocity', 'xml_deviation',
                  'articulation', 'pedal_refresh_time', 'pedal_cut_time',
                  'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                  'pedal_refresh', 'pedal_cut', 'qpm_2', 'beat_dynamic', 'measure_tempo', 'measure_dynamic']
TRILL_LIST_LABELS = [f"trill-{i}" for i in range(5)]
TEST_Y_LABELS = FIRST_TEST_Y_LABELS + TRILL_LIST_LABELS

INPUT_LABELS = TEST_X_LABELS + TEST_Y_LABELS


def get_input_df(test_x, test_y):
    inputs = []
    for i in range(len(test_x)):
        inputs.append(test_x[i] + test_y[i])

    return pd.DataFrame(inputs, columns=INPUT_LABELS)

