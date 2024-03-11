from rich.console import Console

console = Console()


def cp(text="HERE", color="green", level=0):
    tabspace = 4
    if color is not None:
        console.print(" " * (tabspace * level) + f"[{color}]===>[/] " + f"{text}")
    else:
        console.print(text)
