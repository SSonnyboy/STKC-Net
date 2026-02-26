
import copy
import torch


class ModelEMA:
	def __init__(self, model, decay=0.99, device=None):
		self.decay = decay
		self.ema_model = copy.deepcopy(model)
		self.ema_model.eval()
		for param in self.ema_model.parameters():
			param.requires_grad_(False)
		if device is not None:
			self.ema_model.to(device)

	def update(self, model):
		with torch.no_grad():
			msd = model.state_dict()
			for key, value in self.ema_model.state_dict().items():
				if value.dtype.is_floating_point:
					value.copy_(value * self.decay + msd[key].detach() * (1.0 - self.decay))
				else:
					value.copy_(msd[key])

	def state_dict(self):
		return self.ema_model.state_dict()

	def load_state_dict(self, state_dict):
		self.ema_model.load_state_dict(state_dict)

	def to(self, device):
		self.ema_model.to(device)
		return self
