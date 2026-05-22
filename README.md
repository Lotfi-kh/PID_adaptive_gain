# PID_adaptive_gain

Training and evaluation code for the adaptive PID tuning system I developed for
my master's thesis. A small TD3 reinforcement learning policy adjusts the
inner-loop rate PID gains of a quadrotor online, while the PID controller
itself is left in place. The agent only moves the gains.

The platform is an F450 quadrotor. Training runs in PyBullet. Validation runs
in Gazebo with PX4 SITL. The PX4 firmware module that runs the trained policy
on the drone lives in a separate repository.

## The deployed model

The model under `results/frozen_joint_12d_c860k/` is the one used for all
results in the thesis.

- Observation: 12 floats. Roll and pitch, body roll-rate and pitch-rate,
  the two rate errors, and the current normalised gains for both axes
  (Kp, Ki, Kd for roll and for pitch).
- Action: 3 floats in [-1, 1], shared between roll and pitch. They are added
  to the running gains after a fixed scaling: `a[0]*3.4e-2` to Kp,
  `a[1]*3.4e-3` to Ki, `a[2]*1.7e-4` to Kd. A full-scale action of 1.0
  therefore moves a gain by 2% of its allowed range in a single control step.
- Network: Linear(12,64), ReLU, Linear(64,64), ReLU, Linear(64,3), Tanh.
  5187 parameters.
- Gain bounds: Kp in [0, 1.72], Ki in [0, 0.172], Kd in [0, 8.6e-3].
- Trained for 860,000 steps.

On a 48-condition disturbance grid (4 noise levels x 4 magnitudes x 3 axes,
20 episodes each, 960 rollouts total) this policy beats the fixed-gain
baseline in 42 of the 48 conditions. The peak angular rate is reduced by
55.1% on average and the recovery time by 54.2%, with zero crashes across
the 960 rollouts.

## Repo layout

    envs/                              training environment
    train.py                           TD3 trainer
    tests/                             smoke runs and the fixed-gain baseline
    eval_stable_hover.py               hover-time inactivity check
    eval_sustained.py                  sustained constant-torque evaluation
    eval_disturbance.py                transient disturbance evaluation
    run_disturbance_grid.py            48-condition grid runner
    run_aggressive_eval.py             where the headline 42/48 numbers come from
    make_thesis_figs.py                generates the chapter 4 figures
    results/frozen_joint_12d_c860k/    deployed model (.zip, .onnx, summary)
    export/                            ONNX export, test vectors, generated STM32 C code
    wrapper/                           host-side wrapper around the ONNX network
    sitl/                              PX4 SITL helpers and the gain injector

`export/deployment_contract.md` is the reference for the firmware side. It
lists the observation layout, the gain update rule, the clipping bounds and
the safety fallbacks that have to live outside the ONNX graph.

## Install

Python 3.10 or newer.

    pip install -r requirements.txt

PyBullet is CPU bound, so there is no real benefit to running this on GPU.

## Train

    python train.py --env pybullet --steps 1000000

Checkpoints and TensorBoard logs go under `runs/<timestamp>/`. Checkpoints are
written every 10,000 steps and evaluations every 20,000.

## Evaluate the deployed model

    python eval_stable_hover.py \
        --model results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip
    python eval_sustained.py \
        --model results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip
    python run_disturbance_grid.py \
        --model results/frozen_joint_12d_c860k/td3_pid_860000_steps.zip

Each script has its own argparse, so `--help` will list its options.

## Deployment

The ONNX file at `results/frozen_joint_12d_c860k/actor_joint_12d_c860k.onnx`
is what STM32Cube.AI consumes. The generated C code is in
`export/stm32_c_code_12d_c860k/`. Host-side bit-exactness of the ONNX network
against the original PyTorch model can be re-checked with
`export/verify_actor_onnx.py`. The PX4 module that calls this network on the
flight controller is in the companion repository.

## Thesis

Chapter 3 of the thesis describes the method and chapter 4 reports the
results.
