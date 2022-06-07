pip install -r requirements.txt

mkdir ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

kaggle competitions download -c histopathologic-cancer-detection

mkdir dataset
unzip -q histopathologic-cancer-detection.zip -d dataset
rm histopathologic-cancer-detection.zip