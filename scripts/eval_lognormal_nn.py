import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.append("../")
from helper import load_data, preprocess, data_source_release
from src.distnet_torch import DistNetModel, nllh_loss_torch

def main():
    sc_dict = data_source_release.get_sc_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", dest="scenario", required=True,
                        choices=sc_dict.keys())
    parser.add_argument("--num_train_samples", dest="num_train_samples",
                        default=100, type=int)
    parser.add_argument("--fold", dest="fold", required=True, type=int)
    parser.add_argument("--save", dest="save", required=True)
    parser.add_argument("--seed", dest="seed", required=False, default=100,
                        type=int)
    parser.add_argument("--wclim", dest="wclim", required=False, default=60*59,
                        type=int)  # Wall clock limit in seconds
    parser.add_argument("--neurons", dest="neurons", required=False,
                        default=16, type=int)
    parser.add_argument("--layer", dest="layer", required=False, default=2,
                        type=int)
    parser.add_argument("--epochs", dest="epochs", required=False, default=1000,
                        type=int)
    parser.add_argument("--batch_size", dest="batch_size", required=False,
                        default=16, type=int)
    args = parser.parse_args()
    
    # 1) Assertions
    assert 0 <= args.fold <= 9

    # 2) Setup paths and info - fixed for lognormal_nn.floc
    model_name = "lognormal_nn"
    task = "floc"
    save_path = os.path.join(args.save, "%s.%s.%s.%d.%d.pkl" % (args.scenario,
                                                                task, model_name,
                                                                args.fold,
                                                                args.seed))
    if args.num_train_samples != 100:
        save_path += "_%d" % args.num_train_samples
    
    add_info = {"task": task, "scenario": args.scenario,
                "fold": args.fold, "model": model_name, "loaded": False,
                "num_train_samples": args.num_train_samples,
                "seed": args.seed}

    # 3) Load data
    sc_dict = data_source_release.get_sc_dict()
    data_dir = data_source_release.get_data_dir()

    runtimes, features, sat_ls = load_data.\
        get_data(scenario=args.scenario, data_dir=data_dir,
                 sc_dict=sc_dict, retrieve=sc_dict[args.scenario]['use'])

    features = np.array(features)
    runtimes = np.array(runtimes)
    
    # Get CV splits
    print(runtimes.shape, features.shape)
    idx = list(range(runtimes.shape[0]))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    cntr = -1
    for train, valid in kf.split(idx):
        # Reset seed for every instance
        np.random.seed(2)
        cntr += 1
        if cntr != args.fold:
            continue

        X_train = features[train, :]
        X_valid = features[valid, :]

        y_train = runtimes[train]
        y_valid = runtimes[valid]

        X_train, X_valid = preprocess.preprocess_features(X_train, X_valid,
                                                          scal="meanstd")

        print("Evaluating %s, %s, %s on %s, %s" % (args.scenario, model_name,
                                                   task, str(X_train.shape),
                                                   str(y_train.shape)))

        # Prepare flattened data
        X_trn_flat = np.concatenate(
            [[x for i in range(100)] for x in X_train])
        X_vld_flat = np.concatenate(
            [[x for i in range(100)] for x in X_valid])
        y_trn_flat = y_train.flatten().reshape([-1, 1])
        y_vld_flat = y_valid.flatten().reshape([-1, 1])

        # Unfold data
        subset_idx = list(range(100))
        if args.num_train_samples != 100:
            print("Cut data down to %d samples with seed %d" %
                  (args.num_train_samples, args.seed))
            rs = np.random.RandomState(args.seed)
            rs.shuffle(subset_idx)
            subset_idx = subset_idx[:args.num_train_samples]

            # Only shorten data used for training
            X_trn_flat = np.concatenate(
                [[x for i in range(args.num_train_samples)] for x in X_train])
            y_train = y_train[:, subset_idx]
            y_trn_flat = y_train.flatten().reshape([-1, 1])

            X_vld_flat = np.concatenate(
                [[x for i in range(args.num_train_samples)] for x in X_valid])
            y_valid = y_valid[:, subset_idx]
            y_vld_flat = y_valid.flatten().reshape([-1, 1])

        # Min/Max Scale runtimes
        y_max_ = np.max(y_trn_flat)
        y_min_ = 0

        y_trn_flat = (y_trn_flat - y_min_) / y_max_
        y_vld_flat = (y_vld_flat - y_min_) / y_max_

        y_train = (y_train - y_min_) / y_max_
        y_valid = (y_valid - y_min_) / y_max_

        print("X_train:", X_train.shape)
        print("X_valid:", X_valid.shape)
        print("X_train_flat:", X_trn_flat.shape)
        print("X_valid_flat:", X_vld_flat.shape)

        print()

        print("y_train:", y_train.shape)
        print("y_valid:", y_valid.shape)
        print("y_trn_flat:", y_trn_flat.shape)
        print("y_vld_flat:", y_vld_flat.shape)

        # Train log-normal neural network
        retry = 5
        while retry > 0:
            model = DistNetModel(n_input_features=X_train.shape[1], 
                               n_epochs=args.epochs,
                               wc_time_limit=args.wclim, X_valid=X_vld_flat, y_valid=y_vld_flat,)
            model.batch_size = args.batch_size  # Set batch size
            print("Start training log-normal neural network")
            model.train(X_train=X_trn_flat, y_train=y_trn_flat)
            print("Finished training")
            
            tra_pred = model.predict(X_train)
            val_pred = model.predict(X_valid)
            retry -= 1

            if np.isfinite(tra_pred).all():
                print("Training finished successfully")
                retry = -1
            else:
                print("Training failed, retrying...")
                if retry == 0:
                    raise ValueError("Training failed after 5 retries, exiting...")

        # Save results
        dump_res(tra_pred=tra_pred, val_pred=val_pred,
                 save_path=save_path, add_info=add_info)

        break  # Only process the specified fold


def dump_res(save_path, tra_pred, val_pred, add_info):
    with open(save_path, "wb") as fh:
        pickle.dump([tra_pred, val_pred, add_info], fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped to %s" % save_path)

if __name__ == "__main__":
    main()
