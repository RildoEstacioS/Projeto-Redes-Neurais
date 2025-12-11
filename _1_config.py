import os

# Caminho absoluto desta pasta (onde está _1_config.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Pasta onde estão os dados (por exemplo, subpasta "PVS 9" dentro do projeto)
DATAPATH = os.path.join(PROJECT_ROOT, "PVS 9")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATAPATH:", DATAPATH)
