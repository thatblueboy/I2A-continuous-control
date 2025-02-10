import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium.spaces.box
from torch.utils.tensorboard import SummaryWriter

# Define a simple model to predict the next state
class Dreamer(nn.Module):
    '''
    Dreamer for Gymnasium MuJoCo tasks
    '''
    def __init__(self, env, n_future_steps, action_space, observation_space, policy_hidden_layers, dynamics_hidden_layers, dreamer_save_path):
        super(Dreamer, self).__init__()
        self.prediction_horizon = n_future_steps

        action_space = env.action_space.shape[0]
        observation_space = env.observation_space.shape[0]
        
        self.writer = SummaryWriter(dreamer_save_path)

        # self.observation_space
        dreamer_p_output_activation = self._get_dreamer_p_output(env.action_space)
        dreamer_d_output_activation = self._get_dreamer_d_output(env.observation_space)

        self.action_scale = env.action_space.high[0]
         
        #policy

        # self.dreamer_p = nn.Sequential(
        #     nn.Linear(in_features=observation_space, out_features=128, bias=True),
        #     nn.ELU(alpha=1.0),
        #     nn.Linear(in_features=128, out_features=64, bias=True),
        #     nn.ELU(alpha=1.0),
        #     nn.Linear(in_features=64, out_features=action_space, bias=True),

        #     nn.Tanh() if dreamer_p_output_activation == "tanh" else nn.ELU()
        # )

        p_layers = []
        p_layers.append(nn.Linear(in_features=observation_space, out_features=policy_hidden_layers[0]))
        p_layers.append(nn.ELU())
        for l in range(len(policy_hidden_layers)):
            if l == len(policy_hidden_layers) - 1:
                p_layers.append(nn.Linear(policy_hidden_layers[l], action_space))
                nn.Tanh() if dreamer_p_output_activation == "tanh" else nn.ELU()

            else:
                p_layers.append(nn.Linear(policy_hidden_layers[l], policy_hidden_layers[l + 1]))
                p_layers.append(nn.ELU())
        self.dreamer_p = nn.Sequential(*p_layers)
        
        #model of the environment
        # self.dreamer_d = nn.Sequential(
        #     nn.Linear(in_features=action_space+observation_space, out_features=128, bias=True),
        #     nn.ELU(alpha=1.0),
        #     nn.Linear(in_features=128, out_features=64, bias=True),
        #     nn.ELU(alpha=1.0),
        #     nn.Linear(in_features=64, out_features=observation_space, bias=True),
        # )

        d_layers = []
        d_layers.append(nn.Linear(in_features=observation_space+action_space, out_features=dynamics_hidden_layers[0]))
        d_layers.append(nn.ELU())
        for l in range(len(dynamics_hidden_layers)):
            if l == len(dynamics_hidden_layers) - 1:
                d_layers.append(nn.Linear(dynamics_hidden_layers[l], observation_space))

            else:
                d_layers.append(nn.Linear(dynamics_hidden_layers[l], dynamics_hidden_layers[l + 1]))
                d_layers.append(nn.ELU())
        self.dreamer_d = nn.Sequential(*d_layers)

        self.dreamer_optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_metric = nn.MSELoss()
        self.update_count = 0
        # self.env_model_optimizer = optim.Adam(self.dreamer_d.parameters(), lr=1e-3)

        print("dreamer dynamics model", self.dreamer_d)
        print("dreamer policy", self.dreamer_p)

    def _get_dreamer_p_output(self, action_space): #TODO Implement
        # All Gymnasium Mujoco env action spaces are of type (-n, n), so tanh works for all
        # Scale for n is added seperately
        return "tanh"
    
    def _get_dreamer_d_output(self, obs_space): #TODO implement
        return "relu"
    
    def forward(self, state):
        """
        Predict n_future_steps number of observations into the future
        by iteratively rolling out the policy and environment model.
        
        :param state: Initial state (torch tensor of shape [batch_size, observation_space])
        :return: A list of predicted future states
        """

        with torch.no_grad():
            predictions = []
            current_state = torch.from_numpy(state).float()
            for _ in range(self.prediction_horizon):
                # Generate action from the policy model
                action = self.dreamer_p(current_state)*self.action_scale

                # Predict next state using the environment model
                next_state = self.dreamer_d(torch.concatenate([action, current_state]))

                predictions.append(next_state)

                # Update current state
                current_state = next_state

            return torch.concatenate(predictions).numpy()  # Shape: [batch_size, n_future_steps, observation_space]
    
    def update(self, observation_batch, action_batch, reward_batch, next_obs_batch, done_batch):
        """
        Update the policy and environment model based on collected data.

        :param state_batch: Batch of states (torch tensor of shape [batch_size, observation_space])
        :param action_batch: Batch of actions (torch tensor of shape [batch_size, action_space])
        :param reward_batch: Batch of rewards (torch tensor of shape [batch_size, 1])
        :param next_state_batch: Batch of next states (torch tensor of shape [batch_size, observation_space])
        :param done_batch: Batch of done flags (torch tensor of shape [batch_size, 1])
        :return: Total loss for the update step
        """
        action_batch = torch.from_numpy(action_batch)
        observation_batch = torch.from_numpy(observation_batch)
        next_obs_batch = torch.from_numpy(next_obs_batch)
        # Predict next state using the environment model
       
        predicted_next_obs = self.dreamer_d(torch.concatenate([action_batch, observation_batch], dim=-1))
        predicted_actions = self.dreamer_p(observation_batch)*self.action_scale

        # Environment model loss: MSE between predicted and actual next states
        env_model_loss = self.loss_metric(predicted_next_obs, next_obs_batch)
        policy_loss = self.loss_metric(predicted_actions, action_batch)  # Negative reward to maximize
        loss = env_model_loss + policy_loss

        self.dreamer_optimizer.zero_grad()
        loss.backward()
        self.dreamer_optimizer.step()

        # Log metrics to TensorBoard
        self._log_metrics(env_model_loss.item(), policy_loss.item(), 
                         predicted_next_obs, next_obs_batch,
                         predicted_actions, action_batch)

        # Total loss

        return env_model_loss.item(), policy_loss.item()
    
    def _log_metrics(self, env_model_loss, policy_loss, 
                    predicted_next_obs, actual_next_obs,
                    predicted_actions, actual_actions):
        """Log training metrics to TensorBoard"""
        # Increment update counter
        self.update_count += 1

        # Log losses
        self.writer.add_scalar('Loss/Environment_Model', env_model_loss, self.update_count)
        self.writer.add_scalar('Loss/Policy', policy_loss, self.update_count)
        self.writer.add_scalar('Loss/Total', env_model_loss + policy_loss, self.update_count)

        # Log prediction errors
        with torch.no_grad():
            # State prediction metrics
            state_pred_mae = torch.mean(torch.abs(predicted_next_obs - actual_next_obs))
            state_pred_mse = torch.mean((predicted_next_obs - actual_next_obs) ** 2)
            
            # Action prediction metrics
            action_pred_mae = torch.mean(torch.abs(predicted_actions - actual_actions))
            action_pred_mse = torch.mean((predicted_actions - actual_actions) ** 2)

            # Log prediction metrics
            self.writer.add_scalar('Predictions/State_MAE', state_pred_mae, self.update_count)
            self.writer.add_scalar('Predictions/State_MSE', state_pred_mse, self.update_count)
            self.writer.add_scalar('Predictions/Action_MAE', action_pred_mae, self.update_count)
            self.writer.add_scalar('Predictions/Action_MSE', action_pred_mse, self.update_count)

            # Log model statistics
            # for name, param in self.named_parameters():
            #     self.writer.add_histogram(f'Parameters/{name}', param, self.update_count)
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Gradients/{name}', param.grad, self.update_count)
        
# Example dimensions
# state_dim = 17  # Example state dimension (based on the environment)
# action_dim = 8  # Example action dimension
# next_state_dim = 17  # Same as the state dimension
