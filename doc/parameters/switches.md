## Training Parameters for Switches
```python
scenario = 'configs/switches.cfg'
model_weights = 'switches.h5'
depth_radius = 1.0
depth_contrast = 0.5
learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 6,
    'nb_epoch' : 100,
    'steps' : 5000,
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : [1.0, 0.1],
    'alpha_rate' : 0.7,
    'alpha_wait' : 10,
    'gamma' : 0.9,
    'epsilon' : [1.0, 0.1],
    'epsilon_rate' : 0.7,
    'epislon_wait' : 10,
    'nb_tests' : 100,
    'checkpoint' : 1,
    'filename' : 'switches_.h5'
}
training = 'DQN'
```

##Testing Parameters for Switches
```python
scenario = 'configs/switches.cfg'
model_weights = "switches.h5"
depth_radius = 1.0
depth_contrast = 0.5
test_param = {
    'frame_skips' : 6,
    'nb_frames' : 3
}
nb_runs = 1
testing = 'DQN'
```
