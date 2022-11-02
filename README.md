# Human Pose Estimation

The project was tested on Leonhard with Python 3.7.4, PyTorch 1.8.1, CUDA 10.1 and cuDNN 7.6.4. 

For an in depth description, check our [report](mp-report.pdf).

## Recreating final score

1. Clone this repo (directory that you cloned from now on denoted by `${ROOT}`) and create the virtual 
   environment:
   
   
      mkvirtualenv "env-name"

      workon "env-name"


2. Install dependencies using `pip`.

   `pip install -r requirements.txt`

Note: if one works on Leonhard script `create_sym_link.sh` can be used to set the data folder.
### Data augmentation
3. To download the VOC dataset which is needed for occlusion, run the following commands:


    cd ${ROOT}

    wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

    tar --strip-component=1 -xf VOCtrainval_11-May-2012.tar


Then set the `DATASET.VOC` parameter in the `train.yaml` to the location of VOC2012 and set the `DATASET.OCCLUSION` parameter to `True`.
If you do not change the `train.yaml` file, this should have already been taken care of.

### Training
4. To train the model run the following command:


    python scripts/train.py --cfg experiments/h36m/train.yaml

In order to train the model on Leonhard make sure that `module load eth_proxy` is activated and then submit the job with the following command:


    bsub -n 4 -W 100:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python scripts/train.py --cfg experiments/h36m/train.yaml

All parameters that produce the score are already set in `experiments/h36m/train.yaml`

During the training process, checkpoints are created which can be found under `output/both/pose3d_vol/demo/<TIME_STRING>`

In order to resume training, one can change the `MODEL.RESUME` in `train.yaml` to the name of the checkpoint and run the following command:

    python scripts/train.py --cfg experiments/h36m/train.yaml --model_time_str <TIME_STRING>

This, however, is not necessary to reproduce the results.

### Validation
5. To validate the model run the following command and make sure your device supports CUDA:


    python scripts/valid.py --cfg experiments/h36m/test.yaml

Or using Leonhard:

    bsub -n 4 -W 6:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python scripts/valid.py --cfg experiments/h36m/test.yaml

All necessary parameters are set in `test.yaml`

## Miscellaneous

### Data evolution
First, it is necessary to download the following files:
+ bones.npy
+ jointAngleModel_v2.npy
+ staticPose.npy
+ template.npy  

from [here](https://drive.google.com/drive/folders/1MUcR9oBNUpTAJ7YUWdyVLKCQW874FszI) and put them into the folder 
`${ROOT}/resources/constraints`.
To evolve the dataset run `lib/evolution/evolve.py` using the parameters from `train.yaml`:

`python lib/evolution/evolve.py --cfg experiments/h36m/train.yaml`

It is necessary that data augmentation and occlusion parameter are set to False, other evolution parameters
are specified in `lib/evolution/parameter.py`. 

### Workflow
Running an experiment creates a directory with all the respective files using the date as its folder name.
This includes `checkpoint.pth.tar` (last model), `model_best.pth.tar` (model with best validation score), log files as well as the configuration `train.yaml` file.
The config file that gets saved includes the arguments that were used to overwrite it by passing in commandline arguments.

Important commandline arguments:
+ `--cfg`: model backbone defined in the respective yaml file
+ `--train_debug_mode`: `DEBUG_MODE` parameter used to train for 2 iterations with batch size 2 to see if the code finishes without errors
+ `--model_time_str`: load previously trained checkpoint (defined in yaml's `RESUME`) by passing the folder name 

### Leonhard commands
+ `bsubs` add jobs
+ `bjobs` list jobs (`-l` for details)
+ `bbjobs` better bjobs
+ `bkill 0` kill job
+ `bpeek` see output of jobs (`watch "bpeek | tail -n 30"` for continuous output)
+ `module load eth_proxy` allow network connection, necessary for downloading pre-trained models
+ `bqueues` show how much in use the server is
