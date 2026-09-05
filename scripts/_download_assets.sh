cd assets
python _download.py

cd ..

echo "Configuring Path ..."
python ./scripts/update_embodiment_config_path.py
