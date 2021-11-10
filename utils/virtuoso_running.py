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
from virtuosoNet.virtuoso.inference import inference, get_input_from_xml
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
    args.meas_note = net_param.meas_note
    criterion = virtuoso_utils.make_criterion_func(config.train_params.loss_type, device)

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
    model = model.to(device)
    return model, args

def autoencode(model, args):
    return inference(args, model, args.device)

def encode_style(model, args):
    model = virtuoso_utils.load_weight(model, args.checkpoint)
    model.eval()
    score, input, edges, note_locations = get_input_from_xml(args.xml_path, args.composer, args.qpm_primo,
                                                             model.stats['input_keys'], model.stats['graph_keys'],
                                                             model.stats['stats'], args.device)
    return model.encode_style(input, None, edges, note_locations, num_samples=10)
