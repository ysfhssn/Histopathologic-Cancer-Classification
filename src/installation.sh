python3 -m venv env || python -m venv env
source env/bin/activate || ./env/Scripts/activate

pip install -r requirements.txt

mkdir ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

kaggle competitions download -c histopathologic-cancer-detection

mkdir dataset
unzip -q histopathologic-cancer-detection.zip -d dataset
rm histopathologic-cancer-detection.zip
