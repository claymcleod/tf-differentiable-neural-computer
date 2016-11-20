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
        X_train, X_test, y_train, y_test = load_babi(args.datadir, lesson=1)
        #X_train = X_train[:, -2:, :] # for debugging
        #X_test = X_test[:, -2:, :] # for debugging
        print("== DNC ==")
        print()
        machine = DNC(X_train, y_train, X_test, y_test,
                        summary_dir=args.summary_dir, N=18, W=2, R=1,
                        checkpoint_file="checkpoint.cpkt")
        print("== Training ==")
        print()
        machine.train()
