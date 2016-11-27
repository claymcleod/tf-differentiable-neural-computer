import sys
import argparse

from dnc import DNC
from util_data import *

SUPPORTED_TASKS = ["babi", "copy"]
if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("task", type=str)
        parser.add_argument("datadir", type=str, help="Where do you want your datasets to be?")
        parser.add_argument("--summary-dir", type=str, help="Summary directory for tensorboard", default=None)
        args = parser.parse_args()

        args.task = args.task.lower()
        if not args.task in SUPPORTED_TASKS:
            print("Unsupported task: {}".format(args.task))
            sys.exit(1)

        if args.task == 'babi':
            print("== BABI ==")
            download_babi(args.datadir)
            X_train, X_test, y_train, y_test = load_babi(args.datadir, lesson=1)
        elif args.task == "copy":
            print("== COPY ==")
            X_train, X_test, y_train, y_test = make_copy_dataset(args.datadir)

        #X_train = X_train[:, -2:, :] # for debugging
        #X_test = X_test[:, -2:, :] # for debugging
        print("== DNC ==")
        print()
        machine = DNC(X_train, y_train, X_test, y_test,
                        summary_dir=args.summary_dir, N=X_train.shape[1], W=10, R=3,
                        #checkpoint_file="{}.ckpt".format(args.task),
                        optimizer="RMSProp")
        print("== Training ==")
        print()
        machine.train()
