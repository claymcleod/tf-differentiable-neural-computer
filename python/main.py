import argparse
import datautil
from dnc import DNC

if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("datadir", type=str, help="Where do you want your datasets to be?")
        parser.add_argument("--summary-dir", type=str, help="Summary directory for tensorboard", default=None)
        args = parser.parse_args()

        print("== Downloading data ==")
        print()
        datautil.download_babi(args.datadir)
        X, y = datautil.load_babi(args.datadir)
        print("== DNC ==")
        print()
        machine = DNC(X, y, summary_dir=args.summary_dir, N=11, W=10, R=1)
        #machine = DNC(X, y, summary_dir=args.summary_dir)
        print("== Training ==")
        print()
        machine.train()
