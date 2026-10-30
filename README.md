# ForEnt code

Code used to produce the derived files of the ForEnt dataset and to label its entrapments. Every script reads the released dataset folder (the unzipped `common.zip` and `site_N.zip` files, plus `videos/`).

Dataset: [doi:10.5281/zenodo.23802853](https://doi.org/10.5281/zenodo.23802853)

Tested on Ubuntu 22.04 with Python 3.10 and 3.13.

## Install

```
cd ForEnt-code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`video_sync.py` also needs `pip install -r requirements-video.txt`, an NVIDIA GPU and the SAM 2.1 small checkpoint ([sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)). `map.py` needs DUFOMap built from source (Linux; see its README).

## Scripts

Run these per walk, in this order (`<root>` is the dataset folder and `<seq>` a walk such as `site_1_1`):

| Script | Writes |
|---|---|
| `bag.py <recorded bag> <root> <seq>` | `<seq>.mcap` from the recorded ROS 2 Foxy bag, checked message by message |
| `video_sync.py <root> <seq> --sam <checkpoint> --out <seq>_video.yaml` | third-person video offset |
| `time_sync.py <root> <seq> --video <seq>_video.yaml` | `time_sync.yaml` |
| `gaps.py <root> <seq>` | `gaps.csv` and the `recording` block of `sequence.yaml` |
| `map.py <root> <seq> --dufomap <dufomap_run> --config <dufomap>/assets/config.toml` | `map.pcd` |

The recorded bags are not released; `bag.py` shows how the released bags were made from them. `agreement.py <root>` prints the agreement between the two annotators over all walks (Cohen's kappa and temporal IoU). `forent.py` contains the functions shared by the scripts. The trajectories (`trajectory.csv`) were made with Leg-KILO and a per-walk pose graph, as described in the dataset README.

## Labeller

`labeller/label.py` shows a walk's third-person video and joystick above the robot's signals (commanded and measured speed, how far the robot is behind its command, leg activity, attitude) and the candidate entrapments (where the robot is more than 0.7 m behind its command over 3 s, or has rolled over 70 deg). Labels are saved after every change in the format of `annotations/A1.yaml`, and running the same command again continues the file. For the consensus pass, `--compare` shows two annotators' labels and lists where they disagree (missed entrapments, starts or ends more than 1 s apart, types, falls and interventions).

The labelling rules and an example clip of each type are in the dataset: `common/labelling_guide/README.md`.

With the environment from Install active (`~/ForEnt` is the dataset folder):

```
pip install -r requirements-labeller.txt
python labeller/label.py ~/ForEnt site_6_2 --annotator A1 --out ~/labels/site_6_2_A1.yaml
python labeller/label.py ~/ForEnt site_6_2 --annotator consensus --compare A1 A2 --out ~/labels/site_6_2_consensus.yaml
```

In a new terminal, run `cd ForEnt-code` and `source .venv/bin/activate` first.

## External code

| | Version | Licence |
|---|---|---|
| [DUFOMap](https://github.com/KTH-RPL/dufomap) | 9e239dd, default `assets/config.toml` | BSD-3-Clause |
| [SAM 2](https://github.com/facebookresearch/sam2) | 2b90b9f, `sam2.1_hiera_small` | Apache-2.0 |
| [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) | `google/owlv2-base-patch16-ensemble` | Apache-2.0 |
| [Leg-KILO](https://github.com/ouguangjun/Leg-KILO) | f3a5bd2 | MIT |

## Licence

MIT
