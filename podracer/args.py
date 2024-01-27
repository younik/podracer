from functools import singledispatch
from typing import List, Optional


class Namespace:

    exp_name: str = "test"
    "the name of this experiment"
    seed: int = 1
    "seed of the experiment"
    track: bool = True
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "sebulba"
    "the wandb's project name"
    wandb_entity: str = None
    "the entity (team) of wandb's project"
    capture_video: bool = False
    save_model: bool = False
    "whether to save model into the `runs/{run_name}` folder"
    upload_model: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    log_frequency: int = 10
    "the logging frequency of the model performance (in terms of `updates`)"

    # Algorithm specific arguments
    env_id: str = "minigrid:MiniGrid-Empty-16x16-v0"
    "the id of the environment"
    total_timesteps: int = 5000000
    "total timesteps of the experiments"
    learning_rate: float = 2.5e-4
    "the learning rate of the optimizer"
    local_num_envs: int = 4
    "the number of parallel game environments"
    num_actor_threads: int = 2
    "the number of actor threads to use"
    num_steps: int = 128
    "the number of steps to run in each environment per policy rollout"
    anneal_lr: bool = True
    "Toggle learning rate annealing for policy and value networks"
    gamma: float = 0.99
    "the discount factor gamma"
    gae_lambda: float = 0.95
    "the lambda for the general advantage estimation"
    num_minibatches: int = 4
    "the number of mini-batches"
    gradient_accumulation_steps: int = 1
    "the number of gradient accumulation steps before performing an optimization step"
    update_epochs: int = 4
    "the K epochs to update the policy"
    norm_adv: bool = True
    "Toggles advantages normalization"
    clip_coef: float = 0.1
    "the surrogate clipping coefficient"
    ent_coef: float = 0.01
    "coefficient of the entropy"
    vf_coef: float = 0.5
    "coefficient of the value function"
    max_grad_norm: float = 0.5
    "the maximum norm for the gradient clipping"
    channels: List[int] = [16, 32, 32]
    "the channels of the CNN"
    hiddens: List[int] = [256]
    "the hiddens size of the MLP"

    actor_device_ids: List[int] = [0]
    "the device ids that actor workers will use"
    learner_device_ids: List[int] = [0]
    "the device ids that learner workers will use"
    distributed: bool = False
    "whether to use `jax.distributed`"
    concurrency: bool = False
    "whether to run the actor and learner concurrently"

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0
    num_updates: int = 0
    global_learner_decices: Optional[List] = None
    actor_devices: Optional[List] = None
    learner_devices: Optional[List] = None

    action_size: Optional[int] = None

    def dict_repr(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


args = Namespace()

@singledispatch
def printify(x):
    return str(x)

@printify.register
def printify_list(x: list):
    return [printify(i) for i in x]