## QC Usage

```
git clone https://github.com/natalie8210/FruitandVegetableClassification.git
```
Note: May need to add '!' to the beginning depending on the environment.

```
%cd FruitandVegetableClassification
python scripts/qc_status.py
```
This outputs the count of flag vs. passed photos. It also creates qc_metrics.csv which give the metrics of the different quality control measures (e.g. exposure, blur, etc.).

<img width="277" height="218" alt="image" src="https://github.com/user-attachments/assets/05f50e24-e0bf-432f-84df-28e0bb19bc68" />

Once the repo is cloned, you can upload photos to the `raw` folder inside whatever environment you are working in. Only upload subfolders to the `raw` folder, for example, when downlading from OneDrive, keep the folder structure and have all the photos be in their respective folders, so `raw` should be a directory of folders (e.g. 01471890, 01483309, etc.).
