from utils.virtuoso_running import load_model_and_args, autoencode


model, args = load_model_and_args()
autoencode(model, args)