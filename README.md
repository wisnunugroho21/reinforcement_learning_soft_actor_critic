# Soft Actor Critic

Simple code to demonstrate Reinforcement Learning using Soft Actor Critic

## Getting Started

This project is using Pytorch for Deep Learning Framework and Gym for Reinforcement Learning Environment.  
Although it's not required, but i recommend run this project on a PC with GPU and 8 GB Ram

### Prerequisites

Make sure you have installed Pytorch and Gym.  
- Click [here](https://pytorch.org/get-started/locally/) to install pytorch  
- Click [here](https://gym.openai.com/docs/) to install gym

### Installing

Just clone this project into your work folder

```
git clone https://github.com/wisnunugroho21/reinforcement_learning_soft_actor_critic.git
```

## Running the project

After you clone the project, enter the folder and run following script in cmd/terminal :

```
cd reinforcement_learning_ppo_rnd/SAC
python sac_bipedal_pytorch.py
```

## Soft Actor Critic

Soft Actor Critic (SAC) is an algorithm which optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn’t a direct successor to TD3 (having been published roughly concurrently), but it incorporates the clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also winds up benefiting from something like target policy smoothing.

A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum.

## Future development

I really want to adapt this code to Tensorflow  
But I think it's better to wait for Tensorflow 2.0 to be fully released

For now, I focused on how to implement this project on more difficult environment (Atari games, Roboschool, etc)

## Contributing

This project is far from finish. Any fix, contribute, or an idea is very appreciated

## Source

These codes was borrowed from [this blog](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665)
I refactor the code in order to easier to use in other project and easier to understand the concept
