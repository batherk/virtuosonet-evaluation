from utils.virtuoso_running import load_model_and_args, encode_style
from virtuoso.get_features_from_midi import load_xml_and_perf_midi

model, args = load_model_and_args()
#print(load_xml_and_perf_midi(args.xml_path, args.midi_path, args.composer))
style = encode_style(model, args)