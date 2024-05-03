import datetime
import os
import signal
import sys
from pathlib import Path

import yaml
from pyannote.database import get_protocol
from pyannote.database.registry import registry
from pytorch_lightning.utilities.model_summary import get_human_readable_count
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel

from pyannote.audio.utils.params import merge_dict


class Parameters:
    """
    Class meant to store and manipulate experiments
    Args:
        train_params: training parameters
        opti_paras: optimization parameters
        eval_params: evaulation parameters
        console: rich console if already existing
        experiment: Id to the experiment if already created
        yaml_path: Path to the config yaml if already created
    """

    def __init__(
        self,
        train_params={},
        opti_params={},
        eval_params={},
        console=None,
        experiment=None,
        yaml_path=None,
    ):
        date = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        DEFAULT_TRAIN_PARAMETERS = {
            "train": {
                "batch_size": 64,
                "num_workers": 10,
                "loss": "bce",
                "optimizer": "Adam",
                "max_speakers_per_frame": 2,
                "lr": 1e-3,
                "patience": 15,
                "max_epochs": 300,
                "gradient_clip_val": 0.5,
                "date": date,
            },
            "model": {
                "segmentation": "pyannote/segmentation-3.0",
                "pipeline": "pyannote/speaker-diarization-3.1",
            },
            "augmentation": {
                "noise_path": "/gpfsdswork/dataset/MUSAN/noise/free-sound/",
            },
        }
        if console is None:
            self.console = Console()
        else:
            self.console = console

        DEFAULT_OPTI_PARAMETERS = {
            "clu_th": 0.5,
            "min_cluster_size": 12,
        }
        if "SSeRiouSs" in train_params:
            train_params["model"]["segmentation"] = {
                "name": "SSeRiouSs",
                "config": train_params["SSeRiouSs"],
            }
            del train_params["SSeRiouSs"]
        elif "segmentation" in train_params:
            train_params["model"]["segmentation"] = {
                "name": train_params["segmentation"],
                "config": None,
            }

        self.train_parameters = merge_dict(DEFAULT_TRAIN_PARAMETERS, train_params)
        self.experiment_ = experiment
        self.optimize_parameters = merge_dict(DEFAULT_OPTI_PARAMETERS, opti_params)
        self.eval_parameters = eval_params
        self.yaml_path = yaml_path
        signal.signal(signal.SIGUSR1, self.sig_handler)

    @classmethod
    def load_from_yaml(cls, yaml_path, console=None):
        """
        yaml_path: path to the config file to load

        Create a parameter object reading the config file
        """
        with open(yaml_path) as yaml_in:
            cfg = yaml.safe_load(yaml_in)
        train_params = cfg["train"]
        opti_param = cfg.get("optimize", {})
        eval_param = cfg.get("evaluate", {})
        experiment = cfg.get("experiment", None)
        yaml_path = cfg.get("yaml_path", None)
        return cls(
            train_params,
            opti_param,
            eval_param,
            console,
            experiment=experiment,
            yaml_path=yaml_path,
        )

    def sig_handler(self, signum, frame):
        """
        If i remember to call slurm with a special flag it save the model/config before stopping
        """
        self.console.print(f"[red]Signal handler called with signal {signum}[/]")
        prod_id = int(os.environ["SLURM_PROCID"])
        self.console.print("The program will be cleanily interrupted")
        if prod_id == 0:
            self.save()
            raise KeyboardInterrupt()  # Clean stop for lightning
        else:
            pass
        sys.exit(-1)

    def save(self, path=None, overwrite=False):
        """
        Save the config in the yaml_path
        """
        if self.yaml_path is None and path is None:
            raise ValueError(
                "Need to specify the output path for the configuration file at least once"
            )
        if self.yaml_path is None or overwrite:
            self.yaml_path = path

        with open(self.yaml_path, "w") as yaml_out:
            out = {
                "train": self.train_parameters,
                "optimize": self.optimize_parameters,
                "evaluate": self.eval_parameters,
                "experiment": self.experiment_,
                "yaml_path": self.yaml_path,
            }
            yaml.dump(out, yaml_out)
            self.console.print(f"Configuration saved at {self.yaml_path}")

    def set_augment(
        self,
        augment,
    ):
        """
        Save a augmentation pipeline as a dictionnary in the parameters
        """
        ii = 0
        for modules in augment.children():
            if modules.__class__.__name__ == "ModuleList":
                self.train_parameters["augmentation"][f"Compose_{ii}"] = {}
                jj = 0
                for child in modules.children():
                    if child.__class__.__name__ == "OneOf":
                        self.train_parameters["augmentation"][f"Compose_{ii}"][
                            f"OneOf_{jj}"
                        ] = {}

                        for childi in next(
                            child.children()
                        ).children():  # Skip ModuleList
                            if childi.__class__.__name__ == "AddBackgroundNoise":
                                self.train_parameters["augmentation"][f"Compose_{ii}"][
                                    f"OneOf_{jj}"
                                ][f"AddBackgroundNoise_{jj}"] = {
                                    "noise_path": self.augment["noise_path"],
                                    "min_snr_in_db": childi.min_snr_in_db,
                                    "max_snr_in_db": childi.max_snr_in_db,
                                    "p": childi.p,
                                }
                            if childi.__class__.__name__ == "AddColoredNoise":
                                self.train_parameters["augmentation"][f"Compose_{ii}"][
                                    f"OneOf_{jj}"
                                ][f"AddColoredNoise_{jj}"] = {
                                    "min_f_decay": childi.min_f_decay,
                                    "max_f_decay": childi.max_f_decay,
                                    "min_snr_in_db": childi.min_snr_in_db,
                                    "max_snr_in_db": childi.max_snr_in_db,
                                    "p": childi.p,
                                }
                            if childi.__class__.__name__ == "ConversationGeneration":
                                self.train_parameters["augmentation"][f"Compose_{ii}"][
                                    f"OneOf_{jj}"
                                ][f"ConversationGeneration_{jj}"] = {
                                    "ignore_loss": childi.ignore_loss,
                                    "simple_concat": childi.simple_concat,
                                    "range_seg": list(childi.range_seg),
                                    "range_ov": list(childi.range_ov),
                                    "p": childi.p,
                                }
                            if childi.__class__.__name__ == "ReverbAugmentation":
                                self.train_parameters["augmentation"][f"Compose_{ii}"][
                                    f"OneOf_{jj}"
                                ][f"ReverbAugmentation_{jj}"] = {
                                    "width_range": list(childi.width_range),
                                    "depth_range": list(childi.depth_range),
                                    "height_range": list(childi.height_range),
                                    "max_order": childi.max_order,
                                    "absorption": childi.absorption,
                                    "p": childi.p,
                                }
                    if child.__class__.__name__ == "AddBackgroundNoise":
                        self.train_parameters["augmentation"][f"Compose_{ii}"][
                            f"AddBackgroundNoise_{jj}"
                        ] = {
                            "noise_path": self.augment["noise_path"],
                            "min_snr_in_db": child.min_snr_in_db,
                            "max_snr_in_db": child.max_snr_in_db,
                            "p": child.p,
                        }
                    if child.__class__.__name__ == "AddColoredNoise":
                        self.train_parameters["augmentation"][f"Compose_{ii}"][
                            f"AddColoredNoise_{jj}"
                        ] = {
                            "min_f_decay": child.min_f_decay,
                            "max_f_decay": child.max_f_decay,
                            "min_snr_in_db": child.min_snr_in_db,
                            "max_snr_in_db": child.max_snr_in_db,
                            "p": child.p,
                        }
                    if child.__class__.__name__ == "ConversationGeneration":
                        self.train_parameters["augmentation"][f"Compose_{ii}"][
                            f"ConversationGeneration_{jj}"
                        ] = {
                            "ignore_loss": child.ignore_loss,
                            "simple_concat": child.simple_concat,
                            "range_seg": list(child.range_seg),
                            "range_ov": list(child.range_ov),
                            "p": child.p,
                        }
                    if child.__class__.__name__ == "ReverbAugmentation":
                        self.train_parameters["augmentation"][f"Compose_{ii}"][
                            f"ReverbAugmentation_{jj}"
                        ] = {
                            "width_range": list(child.width_range),
                            "depth_range": list(child.depth_range),
                            "height_range": list(child.height_range),
                            "max_order": child.max_order,
                            "absorption": child.absorption,
                            "p": child.p,
                        }

                ii += 1
            else:
                print(modules.__class__().__name__())

    def update_opti(self, params):
        """
        Update the optimization parameters
        """
        self.optimize_parameters = merge_dict(self.optimize_parameters, params)

    def create_experiment(self, name="tmp", force=False):
        """
        Create a new experiment, as well as every folder necessary
        """
        if self.experiment_ is not None and not force:
            print("Experiment already exists, pass force flag to bypass")
            return
        self.experiment_ = f"{name}_{self.train['date']}"
        path = Path("results/" + self.experiment_ + "/models")
        path.mkdir(parents=True, exist_ok=False)
        path_eval = Path("results/" + self.experiment_ + "/eval/rttm")
        path_eval.mkdir(parents=True, exist_ok=False)
        path_eval_audio = Path("results/" + self.experiment_ + "/eval/audio")
        path_eval_audio.mkdir(parents=True, exist_ok=False)
        path_tb = Path("results/" + self.experiment_ + "/tb_logs")
        path_tb.mkdir(parents=True, exist_ok=False)

    @property
    def experiment(self):
        return self.experiment_

    def update_eval(self, params):
        self.eval_parameters = merge_dict(self.eval_parameters, params)

    def set_num_parameters(self, params):
        self.train_parameters["model"]["total_parameters"] = params["total_parameters"]
        self.train_parameters["model"]["trainable_parameters"] = params[
            "trainable_parameters"
        ]

    def set_train_duration(self, time):
        self.train_parameters["train"]["duration"] = time

    def update_duration(self, time):
        if "duration" not in self.train_parameters["train"]:
            self.set_train_duration(0)
        self.train_parameters["train"]["duration"] += time

    def set_score(self, protocol, params):
        """
        Save the best score for a protocol
        """
        try:
            protocol = protocol.name
        except AttributeError:
            pass

        if "score" not in self.eval_parameters:
            self.eval_parameters["score"] = {}
        self.eval_parameters["score"][protocol] = params

    def set_gpu(self, gpu):
        self.train_parameters["train"]["hardware"] = gpu

    @property
    def train(self):
        return self.train_parameters["train"]

    @property
    def opti(self):
        return self.optimize_parameters

    @property
    def model(self):
        return self.train_parameters["model"]

    @property
    def augment(self):
        return self.train_parameters["augmentation"]

    def model_param__str__(self):
        list = []
        # param.add_renderable("[yellow]Model[/]")
        list.append(f"[green]Segmentation[/] : {self.model['segmentation']['name']}")
        if self.model["segmentation"]["config"] is not None:
            for k in self.model["segmentation"]["config"]:
                list.append(
                    f"    [green]{k}[/]:{self.model['segmentation']['config'][k]}"
                )
        list.append(f"[green]Pipeline[/] : {self.model['pipeline']}")
        if "total_parameters" in self.model:
            list.append(
                f"[green]Parameters[/] : {get_human_readable_count(self.model['total_parameters'])}"
            )
            list.append(
                f"    [green]Trainable[/] : {get_human_readable_count(self.model['trainable_parameters'])}"
            )
        else:
            list.append("[red]No information on number of parameters ![/]")
        param = Panel(Group(*list), title="Model", expand=True)
        return param

    def print_model_param(self):
        self.console.print(self.model_param__str__())

    def augment_param__str__(self):
        list = []
        print(self.train_parameters["augmentation"])
        if self.train_parameters["augmentation"] is not None:
            for k in self.train_parameters["augmentation"]:
                if "Compose" in k:
                    for lst in self.train_parameters["augmentation"][k]:
                        if "OneOf" in lst:
                            list.append("  [green]OneOf[/]")
                            for m in self.train_parameters["augmentation"][k][lst]:
                                list.append(f"    [green]{m}[/]")

                                for c in self.train_parameters["augmentation"][k][lst][
                                    m
                                ]:
                                    list.append(
                                        f"      [green]{c}[/] : {self.train_parameters['augmentation'][k][lst][m][c]}"
                                    )
                        else:
                            list.append(f"  [green]{lst}[/]")

                            for c in self.train_parameters["augmentation"][k][lst]:
                                list.append(
                                    f"    [green]{c}[/] : {self.train_parameters['augmentation'][k][lst][c]}"
                                )
                # list.append(f"[green]{k}[/]:{self.train_parameters['augmentation'][k]}")
        else:
            list.append("[green]No Augment[/]")
        param = Panel(Group(*list), title="Augmentation", expand=True)
        return param

    def print_augment_param(self):
        self.console.print(self.augment_param__str__())

    def print_dataset_param(self):
        self.console.print("[yellow]Dataset[/]")
        self.console.print(
            f"[green]Database[/] : {self.train_parameters['dataset']['database']}"
        )
        self.console.print(
            f"[green]Protocol[/] : {self.train_parameters['dataset']['protocol']}"
        )

    def train_param__str__(self, mode="full"):
        list = []
        # param.add_renderable("[yellow]Training parameters[/]")
        if mode == "minimal":
            params = [
                "batch_size",
                "loss",
                "optimizer",
                "lr",
                "best_epoch",
                "duration",
                "hardware",
            ]
            for k in params:
                if k in self.train:
                    if k == "duration":
                        val = str(datetime.timedelta(seconds=self.train[k]))
                    else:
                        val = self.train[k]
                    list.append(f"[green]{k}[/]:{val}")

                else:
                    list.append(f"[red]MISSING INFORMATION ON {k}[/]")
            param = Panel(Group(*list), title="Training parameters", expand=True)
            return param
        for k in self.train:
            list.append(f"[green]{k}[/]:{self.train[k]}")
        param = Panel(Group(*list), title="Training parameters", expand=True)
        return param

    def print_train_param(self, mode="full"):
        self.console.print(self.train_param__str__(mode))

    def print_opti_param(self):
        if self.optimize_parameters is None:
            self.console.print("[red]NO OPTIMIZE PARAMETERS[/]")
        else:

            self.console.print("[yellow]Optimization[/] :")
            self.console.print(
                f"[green]Clustering threshold[/] : {self.optimize_parameters['clu_th']}"
            )
            self.console.print(
                f"[green]Min num per cluster[/] : {self.optimize_parameters['min_cluster_size']}"
            )
            if "seg_th" in self.optimize_parameters:
                self.console.print(
                    f"[green]Segmentation threshold[/] : {self.optimize_parameters['seg_th']}"
                )

    def eval_dataset__str__(self):
        list = []
        for k in self.eval_parameters["dataset"]:
            list.append(f"[green]{k}[/] : {self.eval_parameters['dataset'][k]}")
        dataset = Panel(Group(*list), title="Evaluation dataset", expand=True)
        return dataset

    def print_eval_dataset(self):
        self.console.print(self.eval_dataset__str__())

    def print_num_param(self):
        self.console.print("[yellow]Number of parameters[/] :")
        self.console.print(
            f"[green]Parameters[/] : {get_human_readable_count(self.model['total_parameters'])}"
        )
        self.console.print(
            f"[green]Trainable[/] : {get_human_readable_count(self.model['trainable_parameters'])}"
        )

    def score__str__(self):
        list = []
        # score.add_renderable(f"[yellow]Score[/] :")
        for prot in self.eval_parameters["score"]:
            list.append(f"[green]{prot}[/]")
            for k in self.eval_parameters["score"][prot]:
                list.append(
                    f"    [green]{k}[/] : {self.eval_parameters['score'][prot][k]}"
                )
        score = Panel(Group(*list), title="Score", expand=True)
        return score

    def print_score(self):
        self.console.print(self.score__str__())

    def dataset_stats(self, protocol, part):
        stats = protocol.stats(part)
        total_dur = 0
        for f in getattr(protocol, part)():
            ov = f["annotation"].get_overlap()
            dur = ov.duration()
            total_dur += dur

        return {
            "annotated": stats["annotated"],
            "speech": stats["annotation"],
            "overlap": (total_dur / stats["annotated"]) * 100,
        }

    def dataset_infos(self):

        infos = []
        registry.load_database(self.train_parameters["dataset"]["database"])
        train_protocol = get_protocol(self.train_parameters["dataset"]["protocol"])
        train_stats = self.dataset_stats(train_protocol, "train")
        infos.append("[yellow]Training set[/]")
        infos.append(f"[green]{self.train_parameters['dataset']['protocol']}[/]")
        for k in train_stats:
            infos.append(f"    [green]{k}[/] : {train_stats[k]}")
        if "dataset" in self.eval_parameters:
            registry.load_database(self.eval_parameters["dataset"]["database"])

            infos.append("[yellow]Evaluation set[/]")
            for prot in self.eval_parameters["dataset"]["protocol"]:
                eval_protocol = get_protocol(prot)
                eval_stats = self.dataset_stats(eval_protocol, "test")
                infos.append(f"[green]{prot}[/]")
                for k in eval_stats:
                    infos.append(f"    [green]{k}[/] : {eval_stats[k]}")
        dataset_infos = Panel(Group(*infos), title="Dataset", expand=True)
        return dataset_infos

    def print_interspeech_card(self):
        """
        Output every necessary information for interspeech reproducibility guidelines
        """
        dataset = self.dataset_infos()

        metrics = Columns(
            [
                f"[green]Metric used[/] : {self.eval_parameters['metric']}",
            ],
            title="Metrics",
            expand=True,
        )
        score = self.score__str__()
        train_param = self.train_param__str__(mode="minimal")
        model_param = self.model_param__str__()

        main = Group(dataset, metrics, score, train_param, model_param)
        p = Panel.fit(
            renderable=main,
            title="Interspeech",
            subtitle="Guidelines for Reproducibility",
        )
        self.console.print(p)

    def print_parameters(self):
        self.print_model_param()
        self.print_dataset_param()
        self.print_augment_param()
        self.print_train_param()
