from utils.virtuoso_running import load_model_and_args
from virtuoso.dataset import FeatureCollate, EmotionDataset
from torch.utils.data import DataLoader
from virtuoso.train import get_style_from_emotion_data
import pandas as pd
from utils.data_handling import save_data

model, args = load_model_and_args()

emotion_set = EmotionDataset(args.emotion_data_path, type="train", len_slice=args.len_valid_slice * 2, len_graph_slice=args.len_valid_slice * 2, graph_keys=args.graph_keys,)
emotion_loader = DataLoader(emotion_set, 5, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())

styles = get_style_from_emotion_data(model, emotion_loader, args.device)

style_df = pd.DataFrame(styles)
save_data(style_df, 'style_extraction')