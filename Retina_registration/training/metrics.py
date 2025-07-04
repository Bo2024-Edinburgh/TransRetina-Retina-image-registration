from dataclasses import dataclass
from typing import Tuple, Dict  

@dataclass
class TrainingMetrics:
    """Container for training metrics during model optimization.

    This class tracks various components of training loss and performance
    indicators (e.g., Dice score), which are accumulated over iterations
    and averaged for reporting.
    """
    total_loss: float = 0.0
    dice_score: float = 0.0
    sim_loss: float = 0.0
    smooth_loss: float = 0.0
    keypoint_loss: float = 0.0
    attention_loss: float = 0.0

    def reset(self):
        """Reset all training metrics to zero.
        
        This is typically called at the start of a new epoch.
        """
        self.total_loss = 0.0
        self.dice_score = 0.0
        self.sim_loss = 0.0
        self.smooth_loss = 0.0
        self.keypoint_loss = 0.0
        self.attention_loss = 0.0

    def update(self, metrics: Tuple[float, ...]):
        """Accumulate metric values from the current training iteration.

        Args:
            metrics (Tuple[float, ...]): Tuple containing metric values in the following order:
                (total_loss, dice_score, sim_loss, smooth_loss, keypoint_loss, attention_loss).
        """
        if len(metrics) >= 6:
            loss, dice, sim, smooth, keypoint, attention = metrics[:6]
            self.total_loss += loss
            self.dice_score += dice
            self.sim_loss += sim
            self.smooth_loss += smooth
            self.keypoint_loss += keypoint
            self.attention_loss += attention

    def average(self, count: int) -> Dict[str, float]:
        """Compute average metrics over a given number of samples.

        Args:
            count (int): The number of accumulated samples.

        Returns:
            Dict[str, float]: Dictionary containing average values of all metrics.
        """
        return {
            'total_loss': self.total_loss / count,
            'dice_score': self.dice_score / count,
            'sim_loss': self.sim_loss / count,
            'smooth_loss': self.smooth_loss / count,
            'keypoint_loss': self.keypoint_loss / count,
            'attention_loss': self.attention_loss / count,
        }


@dataclass
class ValidationMetrics:
    """Container for validation metrics including both loss-based and distance-based evaluations.

    This class tracks metrics during model validation, capturing not only loss components
    similar to training, but also spatial error measurements.
    """
    # Loss-based metrics (same as training)
    total_loss: float = 0.0
    dice_score: float = 0.0
    sim_loss: float = 0.0
    smooth_loss: float = 0.0
    keypoint_loss: float = 0.0
    attention_loss: float = 0.0

    # Distance-based metrics (specific to validation)
    avg_error: float = 0.0
    max_error: float = 0.0

    def reset(self):
        """Reset all validation metrics to zero.
        
        Typically called before starting a new validation phase.
        """
        self.total_loss = 0.0
        self.dice_score = 0.0
        self.sim_loss = 0.0
        self.smooth_loss = 0.0
        self.keypoint_loss = 0.0
        self.attention_loss = 0.0
        self.avg_error = 0.0
        self.max_error = 0.0

    def update(self, loss_metrics: Tuple[float, ...], distance_metrics: Tuple[float, float]):
        """Accumulate validation metrics from the current evaluation batch.

        Args:
            loss_metrics (Tuple[float, ...]): Loss-related values in the same order as training metrics.
            distance_metrics (Tuple[float, float]): Tuple of (average_error, maximum_error) for spatial performance.
        """
        if len(loss_metrics) >= 6:
            loss, dice, sim, smooth, keypoint, attention = loss_metrics[:6]
            self.total_loss += loss
            self.dice_score += dice
            self.sim_loss += sim
            self.smooth_loss += smooth
            self.keypoint_loss += keypoint
            self.attention_loss += attention

        avg_err, max_err = distance_metrics
        self.avg_error += avg_err
        self.max_error += max_err

    def average(self, count: int) -> Dict[str, float]:
        """Compute average validation metrics over a given number of samples.

        Args:
            count (int): The number of accumulated samples.

        Returns:
            Dict[str, float]: Dictionary containing average values of all metrics.
        """
        return {
            'total_loss': self.total_loss / count,
            'dice_score': self.dice_score / count,
            'sim_loss': self.sim_loss / count,
            'smooth_loss': self.smooth_loss / count,
            'keypoint_loss': self.keypoint_loss / count,
            'attention_loss': self.attention_loss / count,
            'avg_error': self.avg_error / count,
            'max_error': self.max_error / count,
        }
