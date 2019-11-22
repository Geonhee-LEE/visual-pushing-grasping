# Visual Pushing and Grasping Toolbox

### Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning

<<<<<<< HEAD

=======
>>>>>>> 940c30fb0affdabd40e6f20eafd3838ea093f31c
### Usage

To run UR5 with ROS, run the following:
```shell
roslaunch ur_modern_driver ur5_bringup.launch 
```

For launching moveIt and Rviz, run the following:

```shell
roslaunch total_moveit_config total_moveit_planning_execution.launch
```

In pytorch_ros conda env, run training with real UR5 with ROS like following:
```shell
<<<<<<< HEAD
python ros_main.py --tcp_host_ip 192.168.0.3 --tcp_port 30002 --push_rewards --experience_replay --explore_rate_decay --save_visualizations --grasp_only --load_snapshot --snapshot_file logs/2019-11-04-continue/models/snapshot-000840.reinforcement.pth  --continue_logging --logging_directory logs/2019-11-04-continue/
```


=======
python ros_main.py --tcp_host_ip 192.168.0.3 --tcp_port 30002 --push_rewards --experience_replay --explore_rate_decay --save_visualizations --grasp_only --load_snapshot --snapshot_file logs/2019-11-04-continue/models/snapshot-000440.reinforcement.pth  --continue_logging --logging_directory logs/2019-11-04-continue/
```

>>>>>>> 940c30fb0affdabd40e6f20eafd3838ea093f31c
### Instructions

1. Run V-REP (navigate to your V-REP directory and run `./vrep.sh`). From the main menu, select `File` > `Open scene...`, and open the file `visual-pushing-grasping/simulation/simulation.ttt` from this repository.

1. In another terminal window, run the following (simulation will start in the V-REP window): 

    ```shell
    python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10 \
        --push_rewards --experience_replay --explore_rate_decay \
        --is_testing --test_preset_cases --test_preset_file 'simulation/test-cases/test-10-obj-07.txt' \
        --load_snapshot --snapshot_file 'downloads/vpg-original-sim-pretrained-10-obj.pth' \
        --save_visualizations
    ```

Note: you may get a popup window titled "Dynamics content" in your V-REP window. Select the checkbox and press OK. You will have to do this a total of 3 times before it stops annoying you.

## Training

To train a regular VPG policy from scratch in simulation, first start the simulation environment by running V-REP (navigate to your V-REP directory and run `./vrep.sh`). From the main menu, select `File` > `Open scene...`, and open the file `visual-pushing-grasping/simulation/simulation.ttt`. Then navigate to this repository in another terminal window and run the following:

```shell
python main.py --is_sim --push_rewards --experience_replay --explore_rate_decay --save_visualizations
```

Data collected from each training session (including RGB-D images, camera parameters, heightmaps, actions, rewards, model snapshots, visualizations, etc.) is saved into a directory in the `logs` folder. A training session can be resumed by adding the flags `--load_snapshot` and `--continue_logging`, which then loads the latest model snapshot specified by `--snapshot_file` and transition history from the session directory specified by `--logging_directory`:

```shell
python main.py --is_sim --push_rewards --experience_replay --explore_rate_decay --save_visualizations \
    --load_snapshot --snapshot_file 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE/models/snapshot-backup.reinforcement.pth' \
    --continue_logging --logging_directory 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE' \
```

Various training options can be modified or toggled on/off with different flags (run `python main.py -h` to see all options):

```shell
usage: main.py [-h] [--is_sim] [--obj_mesh_dir OBJ_MESH_DIR]
               [--num_obj NUM_OBJ] [--tcp_host_ip TCP_HOST_IP]
               [--tcp_port TCP_PORT] [--rtc_host_ip RTC_HOST_IP]
               [--rtc_port RTC_PORT]
               [--heightmap_resolution HEIGHTMAP_RESOLUTION]
               [--random_seed RANDOM_SEED] [--method METHOD] [--push_rewards]
               [--future_reward_discount FUTURE_REWARD_DISCOUNT]
               [--experience_replay] [--heuristic_bootstrap]
               [--explore_rate_decay] [--grasp_only] [--is_testing]
               [--max_test_trials MAX_TEST_TRIALS] [--test_preset_cases]
               [--test_preset_file TEST_PRESET_FILE] [--load_snapshot]
               [--snapshot_file SNAPSHOT_FILE] [--continue_logging]
               [--logging_directory LOGGING_DIRECTORY] [--save_visualizations]
```

Results from our baseline comparisons and ablation studies in our [paper](https://arxiv.org/pdf/1803.09956.pdf) can be reproduced using these flags. For example:

* Train reactive policies with pushing and grasping (P+G Reactive); specify `--method` to be `'reactive'`, remove `--push_rewards`, remove `--explore_rate_decay`:

    ```shell
    python main.py --is_sim --method 'reactive' --experience_replay --save_visualizations
    ```

* Train reactive policies with grasping-only (Grasping-only); similar arguments as P+G Reactive above, but add `--grasp_only`:

    ```shell
    python main.py --is_sim --method 'reactive' --experience_replay --grasp_only --save_visualizations
    ```

* Train VPG policies without any rewards for pushing (VPG-noreward); similar arguments as regular VPG, but remove `--push_rewards`:

    ```shell
    python main.py --is_sim --experience_replay --explore_rate_decay --save_visualizations
    ```

* Train shortsighted VPG policies with lower discount factors on future rewards (VPG-myopic); similar arguments as regular VPG, but set `--future_reward_discount` to `0.2`:

    ```shell
    python main.py --is_sim --push_rewards --future_reward_discount 0.2 --experience_replay --explore_rate_decay --save_visualizations
    ```

To plot the performance of a session over training time, run the following:

```shell
python plot.py 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE'
```

Solid lines indicate % grasp success rates (primary metric of performance) and dotted lines indicate % push-then-grasp success rates (secondary metric to measure quality of pushes) over training steps. By default, each point in the plot measures the average performance over the last 200 training steps. The range of the x-axis is from 0 to 2500 training steps. You can easily change these parameters at the top of `plot.py`.

To compare performance between different sessions, you can draw multiple plots at a time:

```shell
python plot.py 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE' 'logs/ANOTHER-SESSION-DIRECTORY-NAME-HERE'
```

## Evaluation

We provide a collection 11 test cases in simulation with adversarial clutter. Each test case consists of a configuration of 3 - 6 objects placed in the workspace in front of the robot. These configurations are manually engineered to reflect challenging picking scenarios, and remain exclusive from the training procedure. Across many of these test cases, objects are laid closely side by side, in positions and orientations that even an optimal grasping policy would have trouble successfully picking up any of the objects without de-cluttering first. As a sanity check, a single isolated object is additionally placed in the workspace separate from the configuration. This is just to ensure that all policies have been sufficiently trained prior to the benchmark (*i.e.* a policy is not ready if fails to grasp the isolated object).

<img src="images/test-cases.jpg" width=100% align="middle" />

The [demo](#a-quick-start-demo-in-simulation) above runs our pre-trained model multiple times (x30) on a single test case. To test your own pre-trained model, simply change the location of `--snapshot_file`:

```shell
python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10 \
    --push_rewards --experience_replay --explore_rate_decay \
    --is_testing --test_preset_cases --test_preset_file 'simulation/test-cases/test-10-obj-07.txt' \
    --load_snapshot --snapshot_file 'YOUR-SNAPSHOT-FILE-HERE' \
    --save_visualizations
```

Data from each test case will be saved into a session directory in the `logs` folder. To report the average testing performance over a session, run the following:

```shell
python evaluate.py --session_directory 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE' --method SPECIFY-METHOD --num_obj_complete N
```

where `SPECIFY-METHOD` can be `reactive` or `reinforcement`, depending on the architecture of your model.

`--num_obj_complete N` defines the number of objects that need to be picked in order to consider the task completed. For example, when evaluating our pre-trained model in the demo test case, `N` should be set to 6: 

```shell
python evaluate.py --session_directory 'logs/YOUR-SESSION-DIRECTORY-NAME-HERE' --method 'reinforcement' --num_obj_complete 6
```

Average performance is measured with three metrics (for all metrics, higher is better):
1. Average % completion rate over all test runs: measures the ability of the policy to finish the task by picking up at least `N` objects without failing consecutively for more than 10 attempts.
1. Average % grasp success rate per completion.
1. Average % action efficiency: describes how succinctly the policy is capable of finishing the task. See our [paper](https://arxiv.org/pdf/1803.09956.pdf) for more details on how this is computed.

### Creating Your Own Test Cases in Simulation

To design your own challenging test case:

1. Open the simulation environment in V-REP (navigate to your V-REP directory and run `./vrep.sh`). From the main menu, select `File` > `Open scene...`, and open the file `visual-pushing-grasping/simulation/simulation.ttt`.
1. In another terminal window, navigate to this repository and run the following:

    ```shell
    python create.py
    ```

1. In the V-REP window, use the V-REP toolbar (object shift/rotate) to move around objects to desired positions and orientations.
1. In the terminal window type in the name of the text file for which to save the test case, then press enter.
1. Try it out: run a trained model on the test case by running `main.py` just as in the demo, but with the flag `--test_preset_file` pointing to the location of your test case text file.

## Running on a Real Robot (UR5)

The same code in this repository can be used to train on a real UR5 robot arm (tested with UR Software version 1.8). To communicate with later versions of UR software, several minor changes may be necessary in `robot.py` (*e.g.* functions like `parse_tcp_state_data`). Tested with Python 2.7 (not fully tested with Python 3).

### Setting Up Camera System


#### Installation Instructions:

1. Connect your RealSense camera with a USB 3.0 compliant cable (important: RealSense D400 series uses a USB-C cable, but still requires them to be 3.0 compliant to be able to stream RGB-D data).
1. To start the TCP server and RGB-D streaming, run the following:

    ```shell
    ./realsense
    ```

Keep the executable running while calibrating or training with the real robot (instructions below). To test a python TCP client that fetches RGB-D data from the active TCP server, run the following:

```shell
cd visual-pushing-grasping/real
python capture.py
```

#### Instructions:

1. Predefined 3D locations are sampled from a 3D grid of points in the robot's workspace. To modify these locations, change the variables `workspace_limits` and `calib_grid_step` at the top of `calibrate.py`.

1. Measure the offset between the midpoint of the checkerboard pattern to the tool center point in robot coordinates (variable `checkerboard_offset_from_tool`). This offset can change depending on the orientation of the tool (variable `tool_orientation`) as it moves across the predefined locations. Change both of these variables respectively at the top of `calibrate.py`. 

1. The code directly communicates with the robot via TCP. At the top of `calibrate.py`, change variable `tcp_host_ip` to point to the network IP address of your UR5 robot controller.

1. With caution, run the following to move the robot and calibrate:

    ```shell
    python calibrate.py
    ```

The script also optimizes for a z-scale factor and saves it into `real/camera_depth_scale.txt`. This scale factor should be multiplied with each depth pixel captured from the camera. This step is more relevant for the RealSense SR300 cameras, which commonly suffer from a severe scaling problem where the 3D data is often 15-20% smaller than real world coordinates. The D400 series are less likely to have such a severe scaling problem. 

### Training with real robot

To train on the real robot, simply run:

```shell
python main.py --tcp_host_ip 192.168.0.3 --tcp_port 30002 --push_rewards --experience_replay --explore_rate_decay --save_visualizations
```

<<<<<<< HEAD
#### Finetune for continuing training
```shell
python main.py --tcp_host_ip 192.168.0.3 --tcp_port 30002 --push_rewards --experience_replay --explore_rate_decay --save_visualizations --grasp_only --load_snapshot --snapshot_file logs/2019-11-04.13\:18\:01/models/snapshot-000340.reinforcement.pth --continue_logging --logging_directory logs/2019-11-04.13\:18\:01/
```
=======
#### Finetuning 
To continue training with origin weight and logging file, simply run:
```shell
python main.py --tcp_host_ip 192.168.0.3 --tcp_port 30002 --push_rewards --experience_replay --explore_rate_decay --save_visualizations --grasp_only --load_snapshot --snapshot_file logs/2019-11-04.13\:18\:01/models/snapshot-000340.reinforcement.pth --continue_logging --logging_directory logs/2019-11-04.13\:18\:01/
```

>>>>>>> 940c30fb0affdabd40e6f20eafd3838ea093f31c

### Additional Tools

* Use `touch.py` to test calibrated camera extrinsics -- provides a UI where the user can click a point on the RGB-D image, and the robot moves its end-effector to the 3D location of that point
* Use `debug.py` to test robot communication and primitive actions
