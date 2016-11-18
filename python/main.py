import argparse

from dnc import DNC
from util_data import *

if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("datadir", type=str, help="Where do you want your datasets to be?")
        parser.add_argument("--summary-dir", type=str, help="Summary directory for tensorboard", default=None)
        args = parser.parse_args()

        print("== Downloading data ==")
        print()
        download_babi(args.datadir)
        X, y = load_babi(args.datadir)
        #X = X[:, -1:, :] # for debugging
        print("== DNC ==")
        print()
        machine = DNC(X, y, summary_dir=args.summary_dir, N=11, W=10, R=1, checkpoint_file="checkpoint.cpkt")
        print("== Training ==")
        print()
        machine.train()
