# also requires
pip install --target=/home/tgrogers-raid/a/tgrogers/python-package/ pandas
pip install --target=/home/tgrogers-raid/a/tgrogers/python-package/ plotly
wget https://github.com/plotly/orca/releases/download/v1.3.1/orca-1.3.1.AppImage
chmod u+x orca-1.3.1.AppImage
cp orca-1.3.1.AppImage ~/bin
ln -s ~/bin/orca-1.3.1.AppImage ~/bin/orca
export PYTHONPATH=/home/tgrogers-raid/a/tgrogers/python-package/:$PYTHONPATH
export PATH=/home/tgrogers-raid/a/tgrogers/bin/:$PATH
