import pickle
from pathlib import Path

from idmodels.sarix import SARIXModel


def main():
    pkl_dir = Path("/Users/cornell/IdeaProjects/operational-models/covid_gbqr/")
    # pkl_dir = Path('/Users/cornell/IdeaProjects/operational-models/covid_ar6_pooled/')
    pkl_mc_rc_pairs = [
        # ('2025-08-02-model_config.pkl', '2025-08-02-run_config.pkl'),  # works -> 2025-08-02-UMass-gbqr.csv
        ("2025-08-09-model_config.pkl", "2025-08-09-run_config.pkl"),  # fails -> 2025-08-09-UMass-gbqr.csv
    ]
    for model_config_file_name, run_config_file_name in pkl_mc_rc_pairs:
        with open(pkl_dir / model_config_file_name, "rb") as mc_fp, open(pkl_dir / run_config_file_name, "rb") as rc_fp:
            model_config = pickle.load(mc_fp)
            run_config = pickle.load(rc_fp)
        print("*", model_config_file_name, ",", run_config_file_name)
        print("yy", model_config)
        model_config.x = []
        model = SARIXModel(model_config)
        model.run(run_config)


if __name__ == "__main__":
    main()
