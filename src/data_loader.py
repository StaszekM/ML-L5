import pandas as pd


def load_seeds():
    df = pd.read_csv(
        "./seeds_dataset.txt",
        sep="\s+",
        names=[
            "area",
            "perimeter",
            "compactness",
            "length_of_kernel",
            "width_of_kernel",
            "asymmetry_coefficient",
            "kernel_groove_length",
            "class",
        ],
    )

    return df.drop(columns=["class"]), df[["class"]]


def load_glass():
    df = pd.read_csv(
        "./glass.data",
        sep=",",
        names=["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"],
    ).drop(columns=["id"])

    return df.drop(columns=["Type"]), df[["Type"]]
