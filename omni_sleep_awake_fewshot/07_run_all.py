
import argparse

import subprocess

import sys

from pathlib import Path



import pandas as pd





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--episode_table",

        required=True,

    )



    parser.add_argument(

        "--temperature",

        type=float,

        default=1.0,

    )



    parser.add_argument(

        "--rerun",

        action="store_true",

    )



    args = parser.parse_args()



    episode_table = Path(

        args.episode_table

    )



    if not episode_table.exists():

        raise FileNotFoundError(

            episode_table

        )



    episodes = pd.read_csv(

        episode_table

    )



    script = (

        Path(__file__).resolve().parent

        / "06_run_prototype_episode.py"

    )



    if not script.exists():

        raise FileNotFoundError(script)



    completed = 0

    skipped = 0



    for index, row in episodes.iterrows():

        result_dir = Path(

            str(row["result_dir"])

        )



        metrics_path = (

            result_dir / "metrics.csv"

        )



        if (

            metrics_path.exists()

            and not args.rerun

        ):

            print(

                "[{}/{}] Skip seed={} | {}-shot"

                .format(

                    index + 1,

                    len(episodes),

                    int(row["seed"]),

                    int(row["shot"]),

                )

            )



            skipped += 1

            continue



        command = [

            sys.executable,

            str(script),

            "--support_csv",

            str(row["support_csv"]),

            "--query_csv",

            str(row["query_csv"]),

            "--out_dir",

            str(result_dir),

            "--shot",

            str(int(row["shot"])),

            "--seed",

            str(int(row["seed"])),

            "--temperature",

            str(args.temperature),

        ]



        print("\n" + "=" * 90)



        print(

            "[{}/{}] Seed={} | {}-shot"

            .format(

                index + 1,

                len(episodes),

                int(row["seed"]),

                int(row["shot"]),

            )

        )



        print("=" * 90)



        subprocess.run(

            command,

            check=True,

        )



        completed += 1



    print("\nAll experiments completed.")

    print("Completed:", completed)

    print("Skipped:", skipped)





if __name__ == "__main__":

    main()

