from utils.virtuoso_running import load_model_and_args, autoencode_score_to_midi
from utils.data_handling import load_data
from numpy import logical_and
from functools import reduce
from utils.conversion import convert_latent_to_model_z

latents_df = load_data('styles')

composer = 'Bach'
piece = 'french-suite_bwv812_no1_allemande'
slice = 'mm_1-end'
player = 's004'
emotion = 'E1'

composer_mask = latents_df['composer'] == composer
piece_mask = latents_df['piece'] == piece
slice_mask = latents_df['slice'] == slice
player_mask = latents_df['player'] == player
emotion_mask = latents_df['style_name'] == emotion

mask = reduce(logical_and,[composer_mask, piece_mask, slice_mask, player_mask, emotion_mask])
style = latents_df[mask]

latent_mean = style.loc[:, 'l0':].mean().tolist()
initial_z = convert_latent_to_model_z(latent_mean)

model, args = load_model_and_args()
autoencode_score_to_midi(model, args, initial_z)