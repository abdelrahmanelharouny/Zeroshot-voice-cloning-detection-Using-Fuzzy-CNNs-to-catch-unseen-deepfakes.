"""
Advanced Neuro-Fuzzy Inference System (NFIS) for deepfake detection.

This module implements a differentiable Fuzzy Inference System (FIS) that:
- Enhances feature representation from CNN
- Learns fuzzy rules automatically
- Models uncertainty explicitly
- Provides interpretable outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class NeuroFuzzyInferenceSystem(nn.Module):
    """
    Advanced Neuro-Fuzzy Inference System (NFIS) layer.
    
    This layer implements a complete fuzzy inference system with:
    1. Fuzzification Layer: Multiple Gaussian membership functions per feature
    2. Rule Base: Learnable fuzzy rules aggregating memberships
    3. Inference Engine: Rule strength computation with normalization
    4. Defuzzification: Mapping rule activations back to feature space
    5. Uncertainty Estimation: Entropy-based uncertainty scoring
    6. Interpretability: Rule explanation capabilities
    
    Attributes:
        feature_dim (int): Dimension of input features.
        num_memberships (int): Number of membership functions per feature.
        num_rules (int): Number of fuzzy rules in the rule base.
        centers (nn.Parameter): Learnable centers for Gaussian membership functions.
        sigmas (nn.Parameter): Learnable sigmas for Gaussian membership functions.
        rule_weights (nn.Parameter): Learnable weights for rule aggregation.
        rule_to_feature (nn.Parameter): Learnable mapping from rules to feature space.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_memberships: int = 3,
        num_rules: int = 32,
        sigma_init: float = 0.5,
    ) -> None:
        """
        Initialize the Neuro-Fuzzy Inference System.
        
        Args:
            feature_dim: Dimension of input features (default: 128).
            num_memberships: Number of Gaussian membership functions per feature (default: 3).
            num_rules: Number of fuzzy rules in the rule base (default: 32).
            sigma_init: Initial value for sigma parameters (default: 0.5).
        """
        super(NeuroFuzzyInferenceSystem, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_memberships = num_memberships
        self.num_rules = num_rules
        self.sigma_init = sigma_init
        
        # =====================================================================
        # 1. FUZZIFICATION LAYER PARAMETERS
        # Multiple Gaussian membership functions per feature
        # =====================================================================
        # centers: (feature_dim, num_memberships)
        self.centers = nn.Parameter(torch.Tensor(feature_dim, num_memberships))
        # sigmas: (feature_dim, num_memberships)
        self.sigmas = nn.Parameter(torch.Tensor(feature_dim, num_memberships))
        
        # =====================================================================
        # 2. RULE BASE PARAMETERS
        # Learnable combinations of feature memberships
        # =====================================================================
        # rule_weights: (num_rules, feature_dim, num_memberships)
        self.rule_weights = nn.Parameter(torch.Tensor(num_rules, feature_dim, num_memberships))
        
        # =====================================================================
        # 4. DEFUZZIFICATION PARAMETERS
        # Map rule activations back to feature space
        # =====================================================================
        # rule_to_feature: (num_rules, feature_dim)
        self.rule_to_feature = nn.Parameter(torch.Tensor(num_rules, feature_dim))
        
        # Initialize all parameters
        self._initialize_parameters()
        
        # Small epsilon for numerical stability
        self.eps = 1e-8
    
    def _initialize_parameters(self) -> None:
        """
        Initialize all learnable parameters with appropriate distributions.
        
        - Centers: Uniformly distributed based on expected input range
        - Sigmas: Initialized to sigma_init with small variation
        - Rule weights: Xavier initialization
        - Rule-to-feature mapping: Xavier initialization
        """
        # Initialize centers uniformly across expected feature range
        nn.init.uniform_(self.centers, -1.0, 1.0)
        
        # Initialize sigmas with positive values
        nn.init.constant_(self.sigmas, self.sigma_init)
        # Add small variation
        self.sigmas.data += torch.randn_like(self.sigmas.data) * 0.1
        # Ensure sigmas stay positive
        self.sigmas.data = torch.clamp(self.sigmas.data, min=0.1)
        
        # Initialize rule weights using Xavier initialization
        nn.init.xavier_uniform_(self.rule_weights)
        
        # Initialize rule-to-feature mapping using Xavier initialization
        nn.init.xavier_uniform_(self.rule_to_feature)
    
    def _fuzzification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuzzification Layer: Compute membership degrees for each feature.
        
        Implements multiple Gaussian membership functions per feature:
        mu = exp(- (x - centers)^2 / (2 * sigma^2))
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim).
            
        Returns:
            Membership degrees tensor of shape (batch_size, feature_dim, num_memberships).
        """
        batch_size = x.shape[0]
        
        # Reshape x for broadcasting: (batch_size, feature_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute squared differences: (batch_size, feature_dim, num_memberships)
        diff_squared = (x_expanded - self.centers.unsqueeze(0)) ** 2
        
        # Compute denominator with numerical stability: (feature_dim, num_memberships)
        sigma_squared = self.sigmas ** 2 + self.eps
        denominator = 2.0 * sigma_squared
        
        # Compute Gaussian membership: (batch_size, feature_dim, num_memberships)
        exponent = -diff_squared / denominator
        memberships = torch.exp(exponent)
        
        return memberships
    
    def _rule_aggregation(self, memberships: torch.Tensor) -> torch.Tensor:
        """
        Rule Base: Aggregate memberships using learnable rule weights.
        
        Implements fuzzy rules as learnable combinations of feature memberships.
        Uses product operation (differentiable AND) with log-space computation
        for numerical stability.
        
        Args:
            memberships: Membership degrees of shape (batch_size, feature_dim, num_memberships).
            
        Returns:
            Rule activations of shape (batch_size, num_rules).
        """
        batch_size = memberships.shape[0]
        
        # Apply softmax to rule_weights to ensure they represent valid importance weights
        # Shape: (num_rules, feature_dim, num_memberships)
        rule_weights_normalized = F.softmax(self.rule_weights, dim=-1)
        
        # Weighted membership: (batch_size, num_rules, feature_dim)
        # Multiply memberships by rule weights and sum over memberships
        weighted_memberships = memberships.unsqueeze(1) * rule_weights_normalized.unsqueeze(0)
        # Sum over membership dimension: (batch_size, num_rules, feature_dim)
        weighted_sum = weighted_memberships.sum(dim=-1)
        
        # Use log-sum-exp trick for numerical stability in product operation
        # Product of memberships = exp(sum(log(memberships)))
        # Add epsilon to avoid log(0)
        log_memberships = torch.log(weighted_sum + self.eps)
        
        # Sum log-memberships across features (product in original space)
        # Shape: (batch_size, num_rules)
        log_rule_strengths = log_memberships.sum(dim=-1)
        
        # Convert back to linear space
        rule_activations = torch.exp(log_rule_strengths)
        
        return rule_activations
    
    def _inference_engine(self, rule_activations: torch.Tensor) -> torch.Tensor:
        """
        Inference Engine: Normalize rule activations using softmax.
        
        Ensures numerical stability through log-space computations.
        
        Args:
            rule_activations: Raw rule activations of shape (batch_size, num_rules).
            
        Returns:
            Normalized rule activations of shape (batch_size, num_rules).
        """
        # Apply softmax for normalization with numerical stability
        normalized_rules = F.softmax(rule_activations, dim=-1)
        
        return normalized_rules
    
    def _defuzzification(self, normalized_rules: torch.Tensor) -> torch.Tensor:
        """
        Defuzzification / Feature Projection: Map rule activations to feature space.
        
        Computes: refined_features = rule_activations @ rule_to_feature
        
        Args:
            normalized_rules: Normalized rule activations of shape (batch_size, num_rules).
            
        Returns:
            Refined features of shape (batch_size, feature_dim).
        """
        # Matrix multiplication: (batch_size, num_rules) @ (num_rules, feature_dim)
        # Result: (batch_size, feature_dim)
        refined_features = torch.matmul(normalized_rules, self.rule_to_feature)
        
        return refined_features
    
    def _compute_uncertainty(self, normalized_rules: torch.Tensor) -> torch.Tensor:
        """
        Uncertainty Estimation: Compute uncertainty score using entropy.
        
        Higher entropy indicates more uniform rule activation (higher uncertainty).
        Lower entropy indicates confident rule activation (lower uncertainty).
        
        Args:
            normalized_rules: Normalized rule activations of shape (batch_size, num_rules).
            
        Returns:
            Uncertainty scores of shape (batch_size, 1).
        """
        # Compute entropy: H = -sum(p * log(p))
        # Add epsilon to avoid log(0)
        log_probs = torch.log(normalized_rules + self.eps)
        entropy = -torch.sum(normalized_rules * log_probs, dim=-1, keepdim=True)
        
        # Normalize entropy by maximum possible entropy (log(num_rules))
        max_entropy = torch.log(torch.tensor(float(self.num_rules), device=entropy.device) + self.eps)
        normalized_uncertainty = entropy / (max_entropy + self.eps)
        
        return normalized_uncertainty
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Neuro-Fuzzy Inference System.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim).
            
        Returns:
            Dictionary containing:
                - refined_features: Tensor of shape (batch_size, feature_dim)
                - uncertainty: Tensor of shape (batch_size, 1)
                - rule_activations: Tensor of shape (batch_size, num_rules)
        """
        # 1. Fuzzification: Compute membership degrees
        memberships = self._fuzzification(x)  # (batch_size, feature_dim, num_memberships)
        
        # 2. Rule Aggregation: Compute rule strengths
        rule_activations_raw = self._rule_aggregation(memberships)  # (batch_size, num_rules)
        
        # 3. Inference: Normalize rule activations
        normalized_rules = self._inference_engine(rule_activations_raw)  # (batch_size, num_rules)
        
        # 4. Defuzzification: Map to feature space
        refined_features = self._defuzzification(normalized_rules)  # (batch_size, feature_dim)
        
        # 5. Uncertainty Estimation
        uncertainty = self._compute_uncertainty(normalized_rules)  # (batch_size, 1)
        
        return {
            "refined_features": refined_features,
            "uncertainty": uncertainty,
            "rule_activations": normalized_rules,
        }
    
    def explain_rules(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate interpretable descriptions of fuzzy rules.
        
        Analyzes the learned rule weights to identify:
        - Top contributing features per rule
        - Dominant membership functions for each feature-rule pair
        
        Args:
            top_k: Number of top features to report per rule (default: 5).
            
        Returns:
            Dictionary containing interpretable rule descriptions:
                - rules: List of dictionaries with rule details
                - feature_importance: Global feature importance across rules
        """
        # Get rule weights: (num_rules, feature_dim, num_memberships)
        rule_weights = self.rule_weights.detach().cpu()
        
        # Compute feature importance per rule (sum over memberships)
        feature_importance_per_rule = rule_weights.abs().sum(dim=-1)  # (num_rules, feature_dim)
        
        # Get dominant membership per feature-rule pair
        dominant_memberships = rule_weights.argmax(dim=-1)  # (num_rules, feature_dim)
        
        rules_description = []
        
        for rule_idx in range(self.num_rules):
            # Get top-k features for this rule
            rule_importance = feature_importance_per_rule[rule_idx]
            top_features = torch.topk(rule_importance, k=min(top_k, self.feature_dim))
            
            feature_contributions = []
            for feat_idx, importance in zip(top_features.indices.tolist(), top_features.values.tolist()):
                dom_membership = dominant_memberships[rule_idx, feat_idx].item()
                center_val = self.centers[feat_idx, dom_membership].item()
                sigma_val = self.sigmas[feat_idx, dom_membership].item()
                
                feature_contributions.append({
                    "feature_index": feat_idx,
                    "importance": float(importance),
                    "dominant_membership": dom_membership,
                    "center": float(center_val),
                    "sigma": float(sigma_val),
                })
            
            rules_description.append({
                "rule_index": rule_idx,
                "top_features": feature_contributions,
            })
        
        # Compute global feature importance (average across rules)
        global_feature_importance = feature_importance_per_rule.mean(dim=0)
        top_global_features = torch.topk(global_feature_importance, k=min(top_k, self.feature_dim))
        
        global_top_features = []
        for feat_idx, importance in zip(top_global_features.indices.tolist(), top_global_features.values.tolist()):
            global_top_features.append({
                "feature_index": feat_idx,
                "avg_importance": float(importance),
            })
        
        return {
            "rules": rules_description,
            "feature_importance": global_top_features,
            "num_rules": self.num_rules,
            "num_memberships": self.num_memberships,
            "feature_dim": self.feature_dim,
        }
    
    def get_membership_functions(self) -> Dict[str, torch.Tensor]:
        """
        Get the current membership function parameters.
        
        Returns:
            Dictionary containing:
                - centers: Tensor of shape (feature_dim, num_memberships)
                - sigmas: Tensor of shape (feature_dim, num_memberships)
        """
        return {
            "centers": self.centers.detach().cpu(),
            "sigmas": self.sigmas.detach().cpu(),
        }


# Backward compatibility alias
FuzzyLayer = NeuroFuzzyInferenceSystem
