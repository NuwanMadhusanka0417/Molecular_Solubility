import subprocess

command = [
    "python",
    "GVFA_edge_main.py",
    "--seeds", "42,43,44,45,46",
    "--dims", "2000,5000",
    "--sigma_pi", "0,1",
    "--save_csv"
]

subprocess.run(command)