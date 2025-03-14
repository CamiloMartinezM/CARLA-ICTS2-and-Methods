echo "Activating Carla environment..."
workon carla-ci3p-pgmpy
echo "Deleting old Carla processes..."
ps aux | grep carla | awk '{print $2}' | xargs -r kill -9 
echo "Starting Carla..."
python extract_pp_data.py
