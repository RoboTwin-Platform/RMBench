RMBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RMBENCH_ROOT}"

echo "Installing the necessary packages ..."
python -m pip install -r scripts/requirements.txt

echo "Installing pytorch3d ..."
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

echo "Preparing XPolicyLab ..."
bash "${RMBENCH_ROOT}/scripts/update_xpolicylab.sh"

echo "Installing XPolicyLab in the RMBench environment ..."
python -m pip install -e "${RMBENCH_ROOT}/XPolicyLab"

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
SAPIEN_LOCATION=$(python -m pip show sapien | grep 'Location' | awk '{print $2}')/sapien
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER


echo "Adjusting code in mplib/planner.py ..."
MPLIB_LOCATION=$(python -m pip show mplib | grep 'Location' | awk '{print $2}')/mplib
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

echo "Installing Curobo ..."
cd envs
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
python -m pip install -e . --no-build-isolation
python -m pip install warp-lang==1.12.0
python -m pip install setuptools==69.5.1
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See README.md for more instructions."
