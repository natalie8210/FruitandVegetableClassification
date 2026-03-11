# Fruit and Vegetable Classification

Use for team members 

1. Clone this repository 
2. Go to the project sharepoint cite

https://iowa-my.sharepoint.com/personal/washor_uiowa_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwashor%5Fuiowa%5Fedu%2FDocuments%2FSTAT%5F7400%5FImage%5FSubmission&ct=1773247314226&or=Teams%2DHL&ga=1&LOF=1

2. Select all 19 folders, and click download (will take a few minutes, 589.4 MB)
3. The download will be a zipped file. Unzip the file, and move the 19 folders inside to the data/raw folder in your cloned repo
4. Create a venv
5. Run pip install -r requirements.txt
6. Use notebooks/preprocess.ipynb to preprocess each photo in each foler, and save the output in data/preprocess.
     * hit run all
     * use the full pipeline in the last cell of the notebook and loop through each folder, and each photo. 
8. Use the preprocessed data, and fruit_labels_metadata for binary and multiclass labels, for your classification.

