## Training Parameters for Rigid_Turning
```python
scenario = 'rigid_turning.cfg'
model_weights = None
depth_radius = 1.0
depth_contrast = 0.9
learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 4,
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
    'nb_tests' : 50
}
training = 'DQN'
training_arg = []
```

##Testing Parameters for Rigid_Turning
```python
scenario = 'rigid_turning.cfg'  # also compatible with exit_finding.cfg, rigid_turning_validation.cfg, corridors.cfg, curved_turning.cfg
model_weights = "double_dqlearn_DQNModel_rigid_turning.h5"
depth_radius = 1.0
depth_contrast = 0.9
test_param = {
    'frame_skips' : 4,
    'nb_frames' : 3
}
nb_runs = 1
testing = 'DQN'
test_state_prediction = False
```
