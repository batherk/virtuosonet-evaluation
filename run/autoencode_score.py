from utils.virtuoso_running import load_model_and_args, autoencode_score_to_midi


model, args = load_model_and_args()
autoencode_score_to_midi(model, args)