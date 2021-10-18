# ood-playground

Hello! :vulcan_salute:

Here, you may find a plenty of useful and lots of useless code.

### Setting up the Libraries:

###### 1. Install `brain-segm` module:

```
git clone https://gitlab.com/ira-labs/projects/research/ood-playground.git
cd ood-playground
pip install -e .
```

###### 2. Install `surface-distance`:
```
git clone https://github.com/deepmind/surface-distance.git
pip install surface-distance/
```

Original repository: https://github.com/deepmind/surface-distance

To remove the `DeprecationWarning: 'np.bool' is a deprecated alias
for the builtin 'bool'.` please change the line `49`
in `surface_distance/metrics.py` to

```
    if array.dtype != bool:
```

(Might be already fixed by the time you are reading this.)

### Guide to run the Experiments (WMH dataset old example)

###### 1. Paths (neuro)
- path to WMH data: `/nmnt/x3-hdd/data/da_mri/wmh_ants/`
- **path to experiments**: `/nmnt/x4-hdd/experiments/da_exps/`

Do not forget to use `chmod -R 770 [DIR]` and `chown -R [USER]:trusted [DIR]` to share
permissions to write for the other users.

###### 2. Build-run an experiment

To build and instantly run the experiment use
```
build_run [CONFIG] [EXP_PATH]
```

Example:
```
build_run ~/da-research/configs/experiments/baseline_wmh_unet2d/one2all.config /nmnt/x4-hdd/experiments/da_exps/test/
```

You can pass additional options that could be useful:
- `-max [N]` restrict the maximum number of simultaneously running jobs to N.
- `-g [N]` number of GPUs to use. 0 for CPU computations (could be useful
to debug an exp while all GPUs are unavailable), additionally you should set
 `device='cpu'` in config. (1 is default and utilizes GPU.)

###### 3. Separately build and run an experiment

Actually, `build_run` executes 2 following commands: `dpipe-build` and `qexp`

1. In case, if you want to build tons of experiments, then submit them with `-max`
restriction, you use `dpipe-build` until you done:) then use `qexp` on the top
directory of all previously built experiments.

2. In case, if your experiment has been crashed because of bug in the code, you
could just fix the code and re-submit experiment with `qexp`. Probably you also
need to delete `.lock` file in the experiment folder.
(bug in the code, not config, otherwise you should rebuild experiment)

They have similar syntax:

```
dpipe-build [CONFIG] [EXP_PATH] [OPTIONS: {-g, -max}]
qexp [EXP_PATH] [OPTIONS: {-g, -max}]
```

###### 4. Debugging

All logs are being saved in `~/.cache/cluster-utils/logs`, just `cat` it!

Validation scores, train losses and time profiles could be visualized via `tensorboard`.
```
tensorboard --logdir /nmnt/x4-hdd/experiments/da_exps/ --port=6067
```

###### 5. Working with the remote server

Assuming you've run jupyter (or tensorboard) on the remote server (e.g. `neuro-x5`)
```
jupyter-notebook --no-browser --port=8611
```
Open a terminal on your computer and run
```
ssh -L localhost:8611:localhost:8611 -N neuro-x5
```
Now, if you enter `localhost:8611` in your browser, the jupyter running on the
`neuro-x5` will be opened.

Furthermore, if you want to run PyCharm to modify remote files, you need:

1. `mkdir ~/neuro-xx` (create directory only once)
2. `sshfs neuro-x1: ~/neuro-xx` (mounts home directory from remote server into your local folder)
3. (now you can edit files from remote server via your favorite IDE, etc.)
4. When finished, use `sudo umount ~/neuro-xx`
