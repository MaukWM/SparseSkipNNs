import argparse
import json
import logging
import os.path

import wandb

from training.loop import train

project_name = "dynamic_static_skip_no_skip_experiment_18apr"
wandb.init(project=project_name)


# python sweep.py
def main(args):
    wandb.login()
    # Get config
    name = f"{project_name}.json"
    path = f"training/sweep/configs/{name}"
    api = wandb.Api()
    with open(path) as f:
        configs = json.load(f)
    for i, config in enumerate(configs):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        config['cuda'] = args.cuda
        config['use_wandb'] = args.use_wandb

        config = argparse.Namespace(**config)
        run_name = f"{config.name}"
        if args.use_wandb:
            runs = api.runs(f"maukwm/{project_name}")
            corresponding_runs = [run for run in runs if run.name == run_name]
            if len(corresponding_runs) == 0:
                train(config, project=project_name, run_name=run_name)
                continue
            if len(corresponding_runs) > 1:
                logging.warning(f"Multiple runs with the same run name have been found with run name: {run_name}")
                continue

            corresponding_run = corresponding_runs[0]
            if corresponding_run.state == "crashed" or corresponding_run.state == "failed":
                # train(config, project=project_name, run_name=run_name, resume=True)
                print("Crashed/failed run!")
                continue
            if corresponding_run.state == "finished":
                continue
        else:
            train(config, project=project_name, run_name=run_name, use_wandb=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A sweep script for the experiments of the Sparse DenseNet.\n'
                                                 'This script is mainly for running pre-created configs.')

    parser.add_argument("-cuda", action='store_true')

    parser.add_argument("-use-wandb", action="store_false")

    args = parser.parse_args()

    main(args)
