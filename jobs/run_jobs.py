import os

jobs = []

for job in jobs:
    os.system(f"sbatch {job}")
    print(f'submitted {job}')