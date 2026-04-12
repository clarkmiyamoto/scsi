from abc import ABC, abstractmethod
import torch
from torch import Tensor
from distribution import Distribution

class Interpolant(ABC):
	"""
	Base class for defining a stochastic interpolant
	"""
	def __init__(
		self,
		base: Distribution,
		target: Distribution,
		):
		"""
		Args:
			base: Base distribution p_0 (typically noise).
			target: Target distribution p_1 (data).
		"""
		self.base = base
		self.target = target
	
	@abstractmethod
	def alpha(self, ts: Tensor) -> Tensor:
		"""
		Coefficient for base samples. a(0)=1, a(1)=0.
		Args:
			ts. Shape (batch)
		Returns:
			alpha evaluated at ts. Shape (batch).
		"""
		pass
		
	@abstractmethod
	def alpha_dot(self, ts: Tensor) -> Tensor:
		"""
		Time derivative of alpha. 
		Args:
			ts. Shape (batch)
		Returns:
			da/dt evaluated at ts. Shape (batch)."""
		pass
	
	@abstractmethod
	def beta(self, ts: Tensor) -> Tensor:
		"""
		Coefficient for target samples. b(0)=0, b(1)=1.
		Args:
			ts. Shape (batch)
		Returns:
			beta evaluated at ts. Shape (batch)."""
		pass
	
	@abstractmethod
	def beta_dot(self, ts: Tensor) -> Tensor:
		"""
		Time derivative of beta.
		Args:
			ts. Shape (batch)
		Returns:
			db/dt evaluated at ts. Shape (batch).
		"""
		pass
	
	@abstractmethod
	def gamma(self, ts: Tensor) -> Tensor:
		"""
		Coefficient for noise. g(0)=g(1)=0.
		Args:
			ts. Shape (batch)
		Returns:
			gamma evaluated at ts. Shape (batch).
		"""
		pass

	@abstractmethod
	def gamma_dot(self, ts: Tensor) -> Tensor:
		"""
		Time derivative of gamma.
		Args:
			ts. Shape (batch)
		Returns:
			dg/dt evaluated at ts. Shape (batch).
		"""
		pass
	
	def interpolant(self, 
				    ts: Tensor, 
				    x0s: Tensor, 
				    x1s: Tensor,
				    zs: Tensor) -> Tensor:
		'''
		Evaluates the interpolant for various samples
			I|_{t,x_0,x_1,z} = alpha_t x_0 + beta_t x_1 + gamma_t z
			
		Args:
			ts. Shape (batch).
			x0s. Shape (batch, dim).
			x1s. Shape (batch, dim).
			zs. Shape (batch, dim)
		'''
		alphas = self.alpha(ts).unsqueeze(1) # Shape (batch, 1)
		betas = self.beta(ts).unsqueeze(1) # Shape (batch, 1)
		gammas = self.gamma(ts).unsqueeze(1) # Shape (batch, 1)

		return alphas * x0s + betas * x1s + gammas * zs 
	
	def interpolant_dot(self, 
				    ts: Tensor, 
				    x0s: Tensor, 
				    x1s: Tensor,
				    zs: Tensor) -> Tensor:
		'''
		Evaluates the time-derivative of the interpolant for various samples
			\dot I|_{t,x_0,x_1,z} = \dot alpha_t x_0 + \dot beta_t x_1 + \dot gamma_t z
			
		Args:
			ts. Shape (batch).
			x0s. Shape (batch, dim).
			x1s. Shape (batch, dim).
			zs. Shape (batch, dim)
		'''
		alphas = self.alpha_dot(ts).unsqueeze(1) # Shape (batch, 1)
		betas = self.beta_dot(ts).unsqueeze(1) # Shape (batch, 1)
		gammas = self.gamma_dot(ts).unsqueeze(1) # Shape (batch, 1)
		
		return alphas * x0s + betas * x1s + gammas * zs
		
	def sample(self, batch_size) -> Tensor:
		ts = torch.rand(batch_size) # Shape (batch,1)
		x0s = self.base.sample(batch_size) # Shape (batch, dim)
		x1s = self.target.sample(batch_size) # Shape (batch, dim)
		zs = torch.randn_like(x0s) # Shape (batch, dim)
		
		return ts, x0s, x1s, zs

class LinearInterpolant(Interpolant):
    """Linear interpolant with no stochastic noise (ODE mode).

    alpha_t = 1 - t,  beta_t = t,  gamma_t = 0.
    """
    def alpha(self, ts: Tensor) -> Tensor:
        return 1 - ts
    
    def alpha_dot(self, ts: Tensor) -> Tensor:
        return -torch.ones_like(ts)
    
    def beta(self, ts: Tensor) -> Tensor:
        return ts
    
    def beta_dot(self, ts: Tensor) -> Tensor:
        return torch.ones_like(ts)
    
    def gamma(self, ts: Tensor) -> Tensor:
        return torch.zeros_like(ts)
    
    def gamma_dot(self, ts: Tensor) -> Tensor:
        return torch.zeros_like(ts)


class TrigInterpolant(Interpolant):
    """Linear interpolant with trigonometric noise schedule (SDE mode).

    alpha_t = 1 - t,  beta_t = t,  gamma_t = gamma_0 * t * (1 - t).

    The noise coefficient gamma_t satisfies gamma(0) = gamma(1) = 0 as
    required.  The paper uses gamma_0 = 1 by default, and scales epsilon
    separately during inference.  For the SDE experiments in Section 6.1,
    gamma_0 = 0.05 with epsilon_t = gamma_t was used (Appendix D.2).
    """

    def __init__(self, base: Distribution, target: Distribution,
                 gamma_0: float = 0.05):
        super().__init__(base, target)
        self.gamma_0 = gamma_0

    def alpha(self, ts: Tensor) -> Tensor:
        return 1 - ts

    def alpha_dot(self, ts: Tensor) -> Tensor:
        return -torch.ones_like(ts)

    def beta(self, ts: Tensor) -> Tensor:
        return ts

    def beta_dot(self, ts: Tensor) -> Tensor:
        return torch.ones_like(ts)

    def gamma(self, ts: Tensor) -> Tensor:
        return self.gamma_0 * ts * (1 - ts)

    def gamma_dot(self, ts: Tensor) -> Tensor:
        return self.gamma_0 * (1 - 2 * ts)
