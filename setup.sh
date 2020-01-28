mkdir data
cd data
wget "http://labs.criteo.com/wp-content/uploads/2014/07/criteo_conversion_logs.tar.gz"
tar -xf criteo_conversion_logs.tar.gz
cd ..
python run_dfm.py
