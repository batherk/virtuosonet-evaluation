from utils.virtuoso_running import load_model_and_args
from virtuoso.dataset import FeatureCollate, EmotionDataset
from torch.utils.data import DataLoader
from virtuoso.train import get_style_from_emotion_data
import pandas as pd
from utils.data_handling import save_data

EMOTION_AMOUNT = 5
LATENT_SIZE = 16

model, args = load_model_and_args()

emotion_set = EmotionDataset(args.emotion_data_path, type="train", len_slice=args.len_valid_slice * 2, len_graph_slice=args.len_valid_slice * 2, graph_keys=args.graph_keys,)
emotion_loader = DataLoader(emotion_set, EMOTION_AMOUNT, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())

styles = get_style_from_emotion_data(model, emotion_loader, args.device)

style_df = pd.DataFrame(styles)
style_df['name'] = style_df['perform_path'].apply(lambda x: x.split('/')[-1])

emotion_classes = [f"E{i+1}" for i in range(EMOTION_AMOUNT)]
latent_column_names = [f"l{i}" for i in range(LATENT_SIZE)]
column_names = ['composer', 'piece', 'slice', 'player', 'style_name', 'iteration'] + latent_column_names
new_rows = []

for i in range(len(style_df.index)):
    for j, emotion in enumerate(emotion_classes):
        row = style_df.iloc[i]
        composer, piece, slice_number, player, _, _ = row['name'].split('.')
        latents = row[emotion]
        general_info = [composer, piece, slice_number, player, emotion]
        for k, vector in enumerate(latents):
            special_info = [k] + list(vector)
            new_rows.append(general_info + special_info)

new_df = pd.DataFrame(new_rows, columns=column_names)
save_data(new_df, 'styles')