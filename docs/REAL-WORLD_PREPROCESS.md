# Pre-process
Seer exclusively utilizes the [DROID](https://droid-dataset.github.io/) dataset for pre-training. In this section, we describe the data pre-processing and transformation steps for both the [DROID](https://droid-dataset.github.io/) and [OXE](https://robotics-transformer-x.github.io/) datasets. These transformations convert the RLDS format into a standard dataset format, including .png, .npz, and .h5 files. The transformed dataset is organized as follows: /subset_name/episodes/000000/steps/0000/xxx.jpg (h5).
The pre-processing step also unifies action labels across different subsets. For example, it standardizes all control methods to use the delta end-effector pose control, ensuring consistency in the robot's base and end-effector origin and axes. This carefully designed alignment process minimizes confusion caused by different robots and control methods.
To facilitate this process, we create a new environment, seer_pre, which is specifically used for pre-processing the DROID and OXE datasets into our desired format.


## seer_pre env 
**(1) Env**
```python
conda create -n seer_pre python=3.10
conda activate seer_pre
```
**(2) Move to real_preprocess**
```python
cd ${YOUR_PATH_TO_SEER}/real_preprocess
```
**(3) Third Party Packages**
```python
pip install -r requirements.txt
```
**(4) octo_oxe_data_utils (Optional for DROID, Required for OXE)**
```python
cd octo_oxe_data_utils
python setup.py install
cd ..
```
**(5) Mujoco**
```python
pip install mujoco
```
**(6) Torch**
```python
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```
**(7) Dlimp (Important):**
We try to use multiprocess to process data. However, the dataset.py in Dlimp introduce randomness.
replace the dataset.py in /your_anaconda/envs/seer_pre/lib/python3.10/site-packages/dlimp/dataset.py with the one in [dlimp/dataset.py](../real_preprocess/dlimp/dataset.py)

## Run Instructions
You can download the full DROID dataset (1.7TB) in RLDS format using the following command:
```python
gsutil -m cp -r gs://gresearch/robotics/droid <path_to_your_target_dir>
```
If needed, follow the download instructions provided on the [OXE Github page](https://github.com/google-deepmind/open_x_embodiment).

Preparation
```python
cd ${YOUR_PATH_TO_SEER}/real_preprocess
conda activate seer_pre
```
To process the DROID dataset, set the src_dir and tgt_dir paths. You can adjust the num_worker argument to specify the number of processes to use:
```python
python convert_public_droid_to_h5_per_step.py
```
For processing the Franka subsets in the OXE dataset, update the src_root_dir and tgt_dataset_dir paths. Similarly, adjust the num_worker argument for parallel processing:
```python
python convert_tfds_to_h5_per_step_oxe_franka.py
```
To process other subsets of the OXE dataset (excluding Franka), update the src_root_dir and tgt_dataset_dir paths and set the number of worker processes:
```python
python convert_tfds_to_h5_per_step_oxe_others.py
```
