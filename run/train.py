from datetime import datetime
from utils.virtuoso_running import load_model_and_args
from utils.virtuoso_training import train

NUM_EPOCHS = 10

model, args = load_model_and_args()


name = f"{args.model_type} - {datetime.now().strftime('%y%m%d-%H%M%S')}"
train(args,model, args.device, NUM_EPOCHS, args.criterion, name)

