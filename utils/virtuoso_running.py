import sys
from pathlib import Path

from torch import distributed
import torch

from virtuosoNet.virtuoso import model as modelzoo
from virtuosoNet.virtuoso import utils as virtuoso_utils
from virtuosoNet.virtuoso import encoder_score as encs
from virtuosoNet.virtuoso import encoder_perf as encp
from virtuosoNet.virtuoso import decoder as dec
from virtuosoNet.virtuoso import residual_selector as res
from virtuosoNet.virtuoso.inference import save_model_output_as_midi, get_input_from_xml
import utils.virtuoso_settings as args

def load_model_and_args():
    if "isgn" not in args.model_code:
        args.intermediate_loss = False

    if args.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    if args.yml_path is not None:
        config = virtuoso_utils.read_model_setting(args.yml_path)
        net_param = config.nn_params
    else:
        net_param = torch.load(str(args.checkpoint), map_location='cpu')['network_params']
        args.yml_path = list(Path(args.checkpoint).parent.glob('*.yml'))[0]
        config = virtuoso_utils.read_model_setting(args.yml_path)
    args.graph_keys = net_param.graph_keys
    # args.meas_note = net_param.meas_note
    args.criterion = virtuoso_utils.make_criterion_func(config.train_params.loss_type, device)

    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    model = modelzoo.VirtuosoNet()
    model.score_encoder = getattr(encs, net_param.score_encoder_name)(net_param)
    model.performance_encoder = getattr(encp, net_param.performance_encoder_name)(net_param)
    model.residual_info_selector = getattr(res, net_param.residual_info_selector_name)()
    model.performance_decoder = getattr(dec, net_param.performance_decoder_name)(net_param)
    model.network_params = net_param
    if not args.initialize_model:
        model.load_state_dict(torch.load(str(args.checkpoint), map_location='cpu')['state_dict'])
    model = model.to(device)
    return model, args

def autoencode_score_to_midi(model, args, initial_z=None):
    model = virtuoso_utils.load_weight(model, args.checkpoint)
    model.eval()

    if initial_z == None:
        initial_z = 'zero'
    # load score
    score, input, edges, note_locations = get_input_from_xml(args.xml_path, args.composer, args.qpm_primo,
                                                             model.stats['input_keys'], model.stats['graph_keys'],
                                                             model.stats['stats'], args.device)
    with torch.no_grad():
        outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z=initial_z)
        if args.save_cluster:
            attention_weights = model.score_encoder.get_attention_weights(input, edges, note_locations)
        else:
            attention_weights = None
        # outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z='rand')
    Path(args.output_path).mkdir(exist_ok=True)
    save_path = args.output_path / f"{args.xml_path.parent.stem}_{args.xml_path.stem}_by_{args.model_code}.mid"
    save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'],
                              note_locations,
                              args.velocity_multiplier, args.multi_instruments, args.tempo_clock, args.boolPedal,
                              args.disklavier,
                              clock_interval_in_16th=args.clock_interval_in_16th, save_csv=args.save_csv,
                              save_cluster=args.save_cluster,
                              attention_weights=attention_weights, mod_midi_path=args.mod_midi_path)


