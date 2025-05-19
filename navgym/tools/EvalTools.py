from dataclasses import dataclass, asdict
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.dataset.episode import Episode, EpisodeID
import numpy as np


@dataclass
class GoalPredictorMetrics:
    mean_final_pos_to_goal_dist: float = np.inf  # Average distance from final position to goal
    success_rate_final_pos_to_goal: float = 0.   # Success rate from final position to goal
    mean_oracle_pos_to_goal_dist: float = np.inf # Oracle average distance from position to goal
    success_rate_oracle_pos_to_goal: float = 0.  # Oracle success rate from position to goal
    success_rate_weighted_by_path_length: float = 0.

    @classmethod
    def names(cls):
        return list(asdict(cls()))

    def to_dict(self):
        return asdict(self)


def eval_planning_metrics(
    episodes: list[Episode],
    trajectory_logs: dict[EpisodeID, list[Pose4D]]
):
    # Helper function to compute path length
    def calculate_path_length(trajectory: list[Pose4D]) -> float:
        if len(trajectory) < 2:
            return 0.0
        return sum(
            curr.xy.dist_to(prev.xy)
            for curr, prev in zip(trajectory[1:], trajectory[:-1])
        )

    # Initialize variables for SPL calculation
    spl_values = []
    final_pos_to_goal_dists = []
    oracle_pos_to_goal_dists = []

    for eps in episodes:
        trajectory = trajectory_logs[eps.id]
        
        # Distance from final position to goal
        final_dist = trajectory[-1].xy.dist_to(eps.target_position.xy)
        final_pos_to_goal_dists.append(final_dist)
        
        # Oracle distance
        trajectory_xy = [pose.xy for pose in trajectory]
        oracle_dist = min(p.dist_to(eps.target_position.xy) for p in trajectory_xy)
        oracle_pos_to_goal_dists.append(oracle_dist)
        
        # Calculate SPL (fixing the optimal_length value)
        success = float(final_dist <= 20)
        path_length = calculate_path_length(trajectory)
        optimal_length = eps.target_position.xy.dist_to(eps.start_pose.xy)
        
        # Handle division by zero
        denominator = max(path_length, optimal_length)
        spl = success * optimal_length / denominator if denominator > 0 else 0
        spl_values.append(spl)

    metrics = GoalPredictorMetrics(
        mean_final_pos_to_goal_dist=np.mean(final_pos_to_goal_dists),
        success_rate_final_pos_to_goal=(np.array(final_pos_to_goal_dists) <= 20).mean(),
        mean_oracle_pos_to_goal_dist=np.mean(oracle_pos_to_goal_dists),
        success_rate_oracle_pos_to_goal=(np.array(oracle_pos_to_goal_dists) <= 20).mean(),
        success_rate_weighted_by_path_length=np.mean(spl_values),
    )

    return metrics
