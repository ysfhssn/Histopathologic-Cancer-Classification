!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json

mkdir dataset
unzip -q histopathologic-cancer-detection.zip -d dataset
rm histopathologic-cancer-detection.zip

!kaggle competitions download -c histopathologic-cancer-detection

mkdir dataset
unzip -q histopathologic-cancer-detection.zip -d dataset
rm histopathologic-cancer-detection.zip