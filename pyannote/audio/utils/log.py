from rich.console import Console
import torch
import os

console = Console()


def torch_save(buffer, path, suffix):
    if os.path.isfile(f"{path}_{suffix}.torch"):

        with open(f"{path}_{suffix}.torch", "rb") as fin:
            existing = torch.load(fin)
        try:
            out = torch.concat((existing, buffer.detach().cpu()))
        except:
            print(
                f"Not same shape to save {path}_{suffix}, expected something compatible with {existing.shape} got {buffer.shape}"
            )
    else:
        out = buffer.detach().cpu()
    with open(f"{path}_{suffix}.torch", "wb") as fout:
        torch.save(out, fout)
    print(f"{path}_{suffix} saved")


def cp(text="HERE", color="green", level=0):
    tabspace = 4
    if color is not None:
        console.print(" " * (tabspace * level) + f"[{color}]===>[/] " + f"{text}")
    else:
        console.print(text)
