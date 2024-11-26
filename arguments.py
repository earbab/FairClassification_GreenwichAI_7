import argparse

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--run', type =int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default = 50)
parser.add_argument('--stage_1_epoch', type=int, default = 1)
parser.add_argument('--upsample', type=int, default = 20)
parser.add_argument('--momentum', type=float, default=0.9) 
parser.add_argument('--weight_decay', type=float, default=1e-04) 


args = parser.parse_args()

