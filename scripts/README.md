## QC Usage

```
git clone https://github.com/natalie8210/FruitandVegetableClassification.git
```
Note: May need to add '!' to the beginning depending on the environment.

```
%cd FruitandVegetableClassification
python scripts/qc_status.py --raw_dir path/to/raw/data
```
This outputs the count of flag vs. passed photos. It also creates qc_metrics.csv which give the metrics of the different quality control measures (e.g. exposure, blur, etc.). Input your own path to the raw data subfolders when you run the above code. 
<img width="277" height="218" alt="image" src="https://github.com/user-attachments/assets/05f50e24-e0bf-432f-84df-28e0bb19bc68" />


