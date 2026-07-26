import tarfile
import os
data_dir = "../data"

with tarfile.open("datatar.tar.gz","w:gz") as tar:
    tar.add(data_dir, arcname=os.path.basename(data_dir))
