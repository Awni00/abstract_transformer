import os

jobs = ['object_sorting/object_sorting-ee2-ea2-de2-da2.job',
 'object_sorting/object_sorting-ee2-ea2-de2-da0.job',
 'object_sorting/object_sorting-ee0-ea2-de2-da0.job',
 'object_sorting/object_sorting-ee0-ea4-de4-da0.job',
 'object_sorting/object_sorting-ee2-ea0-de2-da0.job',
 'object_sorting/object_sorting-ee4-ea0-de4-da0.job']

for job in jobs:
    os.system(f"sbatch {job}")
    print(f'submitted {job}')