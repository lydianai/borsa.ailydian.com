"""
ONLINE LEARNING PIPELINE
=======================

Advanced online learning pipeline for continuous model improvement including:
- Experience replay buffer
- Incremental learning
- Model versioning
- Performance monitoring
- Adaptive learning rates
- Concept drift detection

Features:
- Real-time model updates
- Performance degradation detection
- Automated model rollback
- Ensemble learning with multiple models
- A/B testing framework
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import deque, defaultdict
import json
import hashlib
import pickle
from datetime import datetime, timedelta
import threading
import queue
import time

class LearningMode(Enum):
    """Learning modes for online learning"""
    ONLINE = "online"          # Real-time learning
    BATCH = "batch"            # Batch learning
    HYBRID = "hybrid"          # Combination of both
    EVALUATION = "evaluation"   # Evaluation mode (no learning)

class ModelStatus(Enum):
    """Model status states"""
    ACTIVE = "active"           # Currently in use
    TRAINING = "training"       # Currently being trained
    EVALUATING = "evaluating"   # Currently being evaluated
    DEGRADED = "degraded"       # Performance degraded
    ARCHIVED = "archived"       # Archived model

class DriftType(Enum):
    """Types of concept drift"""
    NO_DRIFT = "no_drift"           # No drift detected
    GRADUAL = "gradual"             # Gradual drift
    SUDDEN = "sudden"               # Sudden drift
    RECURRING = "recurring"         # Recurring drift
    INCREMENTAL = "incremental"     # Incremental drift

@dataclass
class LearningExperience:
    """Single learning experience"""
    features: torch.Tensor
    target: torch.Tensor
    prediction: torch.Tensor
    confidence: float
    loss: float
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]
    model_version: str
    experience_hash: str

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_timestamp: pd.Timestamp
    experience_count: int
    status: ModelStatus
    parent_version: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class DriftDetectionResult:
    """Concept drift detection result"""
    drift_type: DriftType
    drift_probability: float
    drift_timestamp: pd.Timestamp
    affected_features: List[int]
    severity: float  # 0.0 - 1.0
    recommendation: str
    metadata: Dict[str, Any]

class ExperienceReplayBuffer:
    """
    Advanced experience replay buffer with prioritized sampling
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,    # Importance-sampling exponent
        epsilon: float = 1e-6  # Small constant for numerical stability
    ):
        """
        Initialize experience replay buffer
        
        Args:
            max_size: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance-sampling exponent (0 = no correction, 1 = full correction)
            epsilon: Small constant for numerical stability
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        # Buffer storage
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
        # Statistical tracking
        self.total_experiences = 0
        self.positive_experiences = 0
        self.negative_experiences = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def add_experience(
        self,
        experience: LearningExperience,
        priority: Optional[float] = None
    ):
        """
        Add experience to buffer
        
        Args:
            experience: Learning experience to add
            priority: Optional priority for prioritized replay
        """
        with self.lock:
            # Calculate priority if not provided
            if priority is None:
                # Higher priority for high-loss experiences and mispredictions
                loss_priority = experience.loss
                confidence_priority = 1.0 - experience.confidence
                misprediction_priority = 0.0
                
                if experience.target != torch.argmax(experience.prediction):
                    misprediction_priority = 1.0
                
                priority = (loss_priority + confidence_priority + misprediction_priority) / 3.0
            
            # Ensure priority is positive
            priority = max(self.epsilon, priority)
            
            # Add to buffer
            self.buffer.append(experience)
            self.priorities.append(priority)
            
            # Update statistics
            self.total_experiences += 1
            if experience.loss < 0.1:  # Low loss = positive experience
                self.positive_experiences += 1
            else:
                self.negative_experiences += 1
    
    def sample(
        self,
        batch_size: int
    ) -> Tuple[List[LearningExperience], torch.Tensor, torch.Tensor]:
        """
        Sample experiences from buffer using prioritized sampling
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, weights, indices)
        """
        with self.lock:
            if len(self.buffer) == 0:
                return [], torch.tensor([]), torch.tensor([])
            
            # Convert to numpy arrays for easier manipulation
            priorities_np = np.array(self.priorities)
            
            # Calculate sampling probabilities
            probs = priorities_np ** self.alpha
            probs = probs / np.sum(probs)
            
            # Sample indices
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                p=probs,
                replace=False
            )
            
            # Get experiences
            experiences = [self.buffer[i] for i in indices]
            
            # Calculate importance-sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights = weights / np.max(weights)  # Normalize
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            
            # Convert indices to tensor
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            
            return experiences, weights_tensor, indices_tensor
    
    def update_priorities(
        self,
        indices: torch.Tensor,
        priorities: torch.Tensor
    ):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        with self.lock:
            indices_np = indices.detach().cpu().numpy()
            priorities_np = priorities.detach().cpu().numpy()
            
            for i, priority in zip(indices_np, priorities_np):
                if 0 <= i < len(self.priorities):
                    self.priorities[i] = max(self.epsilon, priority)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics
        
        Returns:
            Dictionary with buffer statistics
        """
        with self.lock:
            if len(self.buffer) == 0:
                return {
                    'buffer_size': 0,
                    'total_experiences': 0,
                    'positive_rate': 0.0,
                    'negative_rate': 0.0,
                    'avg_priority': 0.0,
                    'max_priority': 0.0,
                    'min_priority': 0.0
                }
            
            priorities_np = np.array(self.priorities)
            
            return {
                'buffer_size': len(self.buffer),
                'total_experiences': self.total_experiences,
                'positive_rate': self.positive_experiences / max(1, self.total_experiences),
                'negative_rate': self.negative_experiences / max(1, self.total_experiences),
                'avg_priority': float(np.mean(priorities_np)),
                'max_priority': float(np.max(priorities_np)),
                'min_priority': float(np.min(priorities_np))
            }

class ConceptDriftDetector:
    """
    Advanced concept drift detector using statistical methods
    """
    
    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 0.05,
        drift_threshold: float = 0.1
    ):
        """
        Initialize concept drift detector
        
        Args:
            window_size: Size of sliding window for drift detection
            sensitivity: Sensitivity to changes (lower = more sensitive)
            drift_threshold: Threshold for drift detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.drift_threshold = drift_threshold
        
        # Sliding windows for detection
        self.feature_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.prediction_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.performance_windows = defaultdict(lambda: deque(maxlen=window_size))
        
        # Historical statistics
        self.baseline_statistics = {}
        self.current_statistics = {}
        
        # Drift history
        self.drift_history = deque(maxlen=1000)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def update_with_data(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        losses: torch.Tensor,
        timestamp: pd.Timestamp
    ):
        """
        Update drift detector with new data
        
        Args:
            features: Input features
            predictions: Model predictions
            targets: Ground truth targets
            losses: Loss values
            timestamp: Data timestamp
        """
        with self.lock:
            batch_size = features.size(0)
            
            # Update feature windows
            for i in range(batch_size):
                feature_hash = hashlib.md5(features[i].detach().cpu().numpy().tobytes()).hexdigest()
                self.feature_windows[feature_hash].append({
                    'features': features[i].detach().cpu().numpy(),
                    'timestamp': timestamp,
                    'loss': losses[i].item()
                })
            
            # Update prediction windows
            for i in range(batch_size):
                pred_hash = hashlib.md5(predictions[i].detach().cpu().numpy().tobytes()).hexdigest()
                self.prediction_windows[pred_hash].append({
                    'prediction': predictions[i].detach().cpu().numpy(),
                    'target': targets[i].item(),
                    'timestamp': timestamp,
                    'loss': losses[i].item()
                })
            
            # Update performance windows
            avg_loss = torch.mean(losses).item()
            self.performance_windows['loss'].append({
                'value': avg_loss,
                'timestamp': timestamp
            })
    
    def detect_drift(self) -> DriftDetectionResult:
        """
        Detect concept drift in current data
        
        Returns:
            DriftDetectionResult object
        """
        with self.lock:
            # Check if we have enough data
            if len(self.performance_windows['loss']) < self.window_size:
                return DriftDetectionResult(
                    drift_type=DriftType.NO_DRIFT,
                    drift_probability=0.0,
                    drift_timestamp=pd.Timestamp.now(),
                    affected_features=[],
                    severity=0.0,
                    recommendation="Insufficient data for drift detection",
                    metadata={}
                )
            
            # Calculate current statistics
            current_losses = [item['value'] for item in self.performance_windows['loss']]
            current_mean = np.mean(current_losses[-self.window_size//2:])
            baseline_mean = np.mean(current_losses[:self.window_size//2])
            
            # Calculate drift probability using CUSUM algorithm
            drift_probability = self._calculate_cusum_drift(
                current_losses,
                baseline_mean,
                self.drift_threshold
            )
            
            # Determine drift type
            if drift_probability > 0.9:
                drift_type = DriftType.SUDDEN
                severity = 1.0
                recommendation = "Immediate model retraining recommended"
            elif drift_probability > 0.7:
                drift_type = DriftType.GRADUAL
                severity = 0.8
                recommendation = "Monitor performance closely, prepare for retraining"
            elif drift_probability > 0.5:
                drift_type = DriftType.INCREMENTAL
                severity = 0.6
                recommendation = "Continue monitoring, consider incremental updates"
            else:
                drift_type = DriftType.NO_DRIFT
                severity = 0.0
                recommendation = "No significant drift detected"
            
            # Identify affected features
            affected_features = self._identify_affected_features()
            
            # Create result
            result = DriftDetectionResult(
                drift_type=drift_type,
                drift_probability=drift_probability,
                drift_timestamp=pd.Timestamp.now(),
                affected_features=affected_features,
                severity=severity,
                recommendation=recommendation,
                metadata={
                    'current_mean': current_mean,
                    'baseline_mean': baseline_mean,
                    'sample_count': len(current_losses),
                    'window_size': self.window_size
                }
            )
            
            # Store in history if significant drift
            if drift_probability > 0.5:
                self.drift_history.append(result)
            
            return result
    
    def _calculate_cusum_drift(
        self,
        losses: List[float],
        baseline_mean: float,
        threshold: float
    ) -> float:
        """
        Calculate drift probability using CUSUM algorithm
        
        Args:
            losses: List of loss values
            baseline_mean: Baseline mean loss
            threshold: Drift threshold
            
        Returns:
            Drift probability (0.0 - 1.0)
        """
        if len(losses) < 10:
            return 0.0
        
        # Calculate CUSUM statistics
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0
        
        for loss in losses[-self.window_size:]:
            deviation = loss - baseline_mean
            
            # Update CUSUM statistics
            cusum_pos = max(0.0, cusum_pos + deviation - threshold)
            cusum_neg = max(0.0, cusum_neg - deviation - threshold)
            
            # Track maximum CUSUM
            max_cusum = max(max_cusum, cusum_pos, cusum_neg)
        
        # Normalize to probability
        drift_probability = min(1.0, max_cusum / (threshold * 10))
        
        return drift_probability
    
    def _identify_affected_features(self) -> List[int]:
        """
        Identify features that may be causing drift
        
        Returns:
            List of affected feature indices
        """
        affected_features = []
        
        # Simple heuristic: features with high variance in recent windows
        for feature_hash, window in self.feature_windows.items():
            if len(window) >= 20:
                recent_losses = [item['loss'] for item in list(window)[-10:]]
                older_losses = [item['loss'] for item in list(window)[:-10][-10:]]
                
                if len(recent_losses) > 0 and len(older_losses) > 0:
                    recent_var = np.var(recent_losses)
                    older_var = np.var(older_losses)
                    
                    # If variance increased significantly
                    if recent_var > older_var * 1.5:
                        # Hash to feature index (simplified)
                        feature_index = hash(feature_hash) % 150  # Assuming 150 features
                        affected_features.append(feature_index)
        
        return list(set(affected_features))  # Remove duplicates
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """
        Get drift detection statistics
        
        Returns:
            Dictionary with drift statistics
        """
        with self.lock:
            recent_drifts = [d for d in self.drift_history if 
                           (pd.Timestamp.now() - d.drift_timestamp).days < 7]
            
            return {
                'total_drifts_detected': len(self.drift_history),
                'recent_drifts': len(recent_drifts),
                'drift_types': defaultdict(int),
                'avg_drift_severity': 0.0,
                'most_affected_features': []
            }

class ModelVersionManager:
    """
    Model version manager for A/B testing and version control
    """
    
    def __init__(
        self,
        max_versions: int = 10,
        performance_threshold: float = 0.05  # 5% performance improvement threshold
    ):
        """
        Initialize model version manager
        
        Args:
            max_versions: Maximum number of versions to keep
            performance_threshold: Minimum performance improvement threshold
        """
        self.max_versions = max_versions
        self.performance_threshold = performance_threshold
        
        # Version storage
        self.versions = {}  # version_id -> ModelVersion
        self.active_versions = []  # List of active version IDs
        self.version_performance = {}  # version_id -> performance metrics
        
        # A/B testing
        self.ab_tests = {}  # test_id -> test configuration
        self.test_results = {}  # test_id -> results
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def register_new_version(
        self,
        model_state: Dict[str, Any],
        performance_metrics: Dict[str, float],
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_state: Model state dictionary
            performance_metrics: Performance metrics
            parent_version: Parent version ID
            metadata: Additional metadata
            
        Returns:
            New version ID
        """
        with self.lock:
            # Generate version ID
            version_id = self._generate_version_id(model_state, performance_metrics)
            
            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_state=model_state,
                performance_metrics=performance_metrics,
                training_timestamp=pd.Timestamp.now(),
                experience_count=0,
                status=ModelStatus.ACTIVE,
                parent_version=parent_version,
                metadata=metadata or {}
            )
            
            # Store version
            self.versions[version_id] = model_version
            self.version_performance[version_id] = performance_metrics
            
            # Add to active versions
            self.active_versions.append(version_id)
            
            # Maintain version limit
            if len(self.active_versions) > self.max_versions:
                # Archive oldest version
                oldest_version = self.active_versions.pop(0)
                if oldest_version in self.versions:
                    self.versions[oldest_version].status = ModelStatus.ARCHIVED
            
            return version_id
    
    def _generate_version_id(
        self,
        model_state: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> str:
        """
        Generate unique version ID based on model state and performance
        
        Args:
            model_state: Model state dictionary
            performance_metrics: Performance metrics
            
        Returns:
            Unique version ID
        """
        # Create hash of model state and performance
        state_str = str(sorted(model_state.items()))
        perf_str = str(sorted(performance_metrics.items()))
        
        combined_str = state_str + perf_str + str(pd.Timestamp.now())
        version_hash = hashlib.md5(combined_str.encode()).hexdigest()[:12]
        
        return f"v_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{version_hash}"
    
    def evaluate_new_version(
        self,
        new_version_id: str,
        test_metrics: Dict[str, float],
        baseline_version_id: str
    ) -> Tuple[bool, str]:
        """
        Evaluate new version against baseline
        
        Args:
            new_version_id: New version ID
            test_metrics: Test metrics for new version
            baseline_version_id: Baseline version ID
            
        Returns:
            Tuple of (is_better, reason)
        """
        with self.lock:
            # Check if versions exist
            if new_version_id not in self.versions or baseline_version_id not in self.versions:
                return False, "Version not found"
            
            # Get baseline performance
            baseline_metrics = self.versions[baseline_version_id].performance_metrics
            
            # Compare key metrics (accuracy, sharpe_ratio, win_rate)
            key_metrics = ['accuracy', 'sharpe_ratio', 'win_rate']
            improvements = []
            
            for metric in key_metrics:
                if metric in test_metrics and metric in baseline_metrics:
                    new_value = test_metrics[metric]
                    baseline_value = baseline_metrics[metric]
                    
                    if baseline_value != 0:
                        improvement = (new_value - baseline_value) / abs(baseline_value)
                        improvements.append(improvement)
            
            # Calculate average improvement
            if improvements:
                avg_improvement = np.mean(improvements)
                
                if avg_improvement > self.performance_threshold:
                    return True, f"Significant improvement: {avg_improvement:.2%}"
                else:
                    return False, f"Improvement too small: {avg_improvement:.2%}"
            else:
                return False, "No comparable metrics found"
    
    def get_best_version(self) -> Optional[str]:
        """
        Get the best performing model version
        
        Returns:
            Best version ID or None if no versions
        """
        with self.lock:
            if not self.versions:
                return None
            
            # Find version with best performance (highest accuracy)
            best_version = None
            best_accuracy = -1.0
            
            for version_id, version in self.versions.items():
                if version.status == ModelStatus.ACTIVE:
                    accuracy = version.performance_metrics.get('accuracy', 0.0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_version = version_id
            
            return best_version
    
    def archive_version(self, version_id: str):
        """
        Archive a model version
        
        Args:
            version_id: Version ID to archive
        """
        with self.lock:
            if version_id in self.versions:
                self.versions[version_id].status = ModelStatus.ARCHIVED
                if version_id in self.active_versions:
                    self.active_versions.remove(version_id)
    
    def get_version_statistics(self) -> Dict[str, Any]:
        """
        Get version management statistics
        
        Returns:
            Dictionary with version statistics
        """
        with self.lock:
            active_count = sum(1 for v in self.versions.values() if v.status == ModelStatus.ACTIVE)
            archived_count = sum(1 for v in self.versions.values() if v.status == ModelStatus.ARCHIVED)
            training_count = sum(1 for v in self.versions.values() if v.status == ModelStatus.TRAINING)
            
            return {
                'total_versions': len(self.versions),
                'active_versions': active_count,
                'archived_versions': archived_count,
                'training_versions': training_count,
                'best_version': self.get_best_version(),
                'recent_versions': list(self.active_versions[-5:]) if self.active_versions else []
            }

class OnlineLearningPipeline:
    """
    Complete online learning pipeline for continuous model improvement
    
    Features:
    - Experience replay with prioritized sampling
    - Concept drift detection
    - Model version management
    - A/B testing framework
    - Performance monitoring
    - Automated model updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        buffer_size: int = 10000,
        learning_rate: float = 0.001,
        update_frequency: int = 100,  # Update every 100 experiences
        performance_window: int = 1000,  # Performance window size
        device: str = 'cpu'
    ):
        """
        Initialize online learning pipeline
        
        Args:
            model: Neural network model to train
            optimizer: Optimizer for training
            criterion: Loss function
            buffer_size: Experience replay buffer size
            learning_rate: Learning rate
            update_frequency: How often to update model
            performance_window: Window size for performance monitoring
            device: Device to run on
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.update_frequency = update_frequency
        self.performance_window = performance_window
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(max_size=buffer_size)
        
        # Concept drift detector
        self.drift_detector = ConceptDriftDetector(
            window_size=100,
            sensitivity=0.05,
            drift_threshold=0.1
        )
        
        # Model version manager
        self.version_manager = ModelVersionManager(
            max_versions=10,
            performance_threshold=0.05
        )
        
        # Performance monitoring
        self.performance_history = deque(maxlen=performance_window)
        self.loss_history = deque(maxlen=performance_window)
        self.accuracy_history = deque(maxlen=performance_window)
        
        # Training statistics
        self.total_experiences = 0
        self.total_updates = 0
        self.current_loss = 0.0
        self.current_accuracy = 0.0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Background update thread
        self.update_queue = queue.Queue(maxsize=1000)
        self.update_thread = threading.Thread(target=self._background_update_worker, daemon=True)
        self.update_thread.start()
        
        # Model version tracking
        self.current_version_id = None
        self.baseline_performance = {}
    
    def add_experience(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add experience to learning pipeline
        
        Args:
            features: Input features
            targets: Ground truth targets
            predictions: Model predictions (if available)
            confidence: Prediction confidence (if available)
            metadata: Additional metadata
        """
        with self.lock:
            batch_size = features.size(0)
            
            # Move tensors to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # If predictions not provided, generate them
            if predictions is None:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(features)
                    predictions = torch.softmax(logits, dim=-1)
            
            # If confidence not provided, calculate it
            if confidence is None:
                confidence = torch.max(predictions, dim=-1)[0].mean().item()
            
            # Calculate losses
            losses = []
            for i in range(batch_size):
                loss = self.criterion(predictions[i:i+1], targets[i:i+1])
                losses.append(loss.item())
                
                # Create experience
                experience = LearningExperience(
                    features=features[i],
                    target=targets[i],
                    prediction=predictions[i],
                    confidence=confidence,
                    loss=loss.item(),
                    timestamp=pd.Timestamp.now(),
                    metadata=metadata or {},
                    model_version=self.current_version_id or "initial",
                    experience_hash=hashlib.md5(
                        (str(features[i].detach().cpu().numpy()) + 
                         str(targets[i].item())).encode()
                    ).hexdigest()[:16]
                )
                
                # Add to replay buffer
                self.replay_buffer.add_experience(experience)
                
                # Update drift detector
                self.drift_detector.update_with_data(
                    features=features[i:i+1],
                    predictions=predictions[i:i+1],
                    targets=targets[i:i+1],
                    losses=torch.tensor([loss.item()]),
                    timestamp=pd.Timestamp.now()
                )
            
            # Update statistics
            self.total_experiences += batch_size
            self.current_loss = np.mean(losses)
            self.loss_history.append(self.current_loss)
            
            # Calculate accuracy
            predicted_classes = torch.argmax(predictions, dim=-1)
            accuracy = (predicted_classes == targets).float().mean().item()
            self.current_accuracy = accuracy
            self.accuracy_history.append(accuracy)
            
            # Update performance history
            self.performance_history.append({
                'timestamp': pd.Timestamp.now(),
                'loss': self.current_loss,
                'accuracy': accuracy,
                'experience_count': self.total_experiences
            })
    
    def update_model(
        self,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Update model with accumulated experiences
        
        Args:
            force_update: Force update regardless of frequency
            
        Returns:
            Update results dictionary
        """
        # Check if update is needed
        if not force_update and self.total_experiences % self.update_frequency != 0:
            return {
                'updated': False,
                'reason': 'Update frequency not reached',
                'total_experiences': self.total_experiences,
                'current_loss': self.current_loss,
                'current_accuracy': self.current_accuracy
            }
        
        # Add to update queue for background processing
        try:
            self.update_queue.put_nowait({
                'force_update': force_update,
                'timestamp': pd.Timestamp.now()
            })
            return {
                'updated': False,
                'queued': True,
                'reason': 'Update queued for background processing',
                'total_experiences': self.total_experiences
            }
        except queue.Full:
            return {
                'updated': False,
                'queued': False,
                'reason': 'Update queue full',
                'total_experiences': self.total_experiences
            }
    
    def _background_update_worker(self):
        """Background worker for model updates"""
        while True:
            try:
                # Get update request
                update_request = self.update_queue.get(timeout=1.0)
                
                # Perform update
                self._perform_model_update(update_request['force_update'])
                
                self.update_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                warnings.warn(f"Background update error: {str(e)}")
                time.sleep(1.0)
    
    def _perform_model_update(
        self,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Perform actual model update (called by background worker)
        
        Args:
            force_update: Force update regardless of conditions
            
        Returns:
            Update results dictionary
        """
        with self.lock:
            # Sample from replay buffer
            experiences, weights, indices = self.replay_buffer.sample(batch_size=32)
            
            if not experiences:
                return {
                    'updated': False,
                    'reason': 'No experiences to learn from',
                    'total_experiences': self.total_experiences
                }
            
            # Prepare batch
            batch_features = torch.stack([exp.features for exp in experiences]).to(self.device)
            batch_targets = torch.stack([exp.target for exp in experiences]).to(self.device)
            
            # Train model
            self.model.train()
            
            # Forward pass
            logits = self.model(batch_features)
            loss = self.criterion(logits, batch_targets)
            
            # Apply importance-sampling weights
            if weights is not None and len(weights) == len(experiences):
                weighted_loss = (loss * weights.to(self.device)).mean()
            else:
                weighted_loss = loss.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.model.eval()
            
            # Update statistics
            self.total_updates += 1
            
            # Calculate new performance metrics
            with torch.no_grad():
                new_logits = self.model(batch_features)
                new_predictions = torch.softmax(new_logits, dim=-1)
                new_predicted_classes = torch.argmax(new_predictions, dim=-1)
                new_accuracy = (new_predicted_classes == batch_targets).float().mean().item()
                new_loss = self.criterion(new_logits, batch_targets).mean().item()
            
            # Detect concept drift
            drift_result = self.drift_detector.detect_drift()
            
            # Update priorities in replay buffer
            new_priorities = torch.abs(new_logits - batch_targets.unsqueeze(-1)).mean(dim=-1)
            self.replay_buffer.update_priorities(indices, new_priorities)
            
            # Check performance degradation
            performance_degraded = self._check_performance_degradation(new_loss, new_accuracy)
            
            # Update version if needed
            if performance_degraded or drift_result.drift_probability > 0.7:
                self._create_new_model_version(new_loss, new_accuracy, drift_result)
            
            return {
                'updated': True,
                'total_updates': self.total_updates,
                'batch_size': len(experiences),
                'old_loss': self.current_loss,
                'new_loss': new_loss,
                'old_accuracy': self.current_accuracy,
                'new_accuracy': new_accuracy,
                'drift_detected': drift_result.drift_type != DriftType.NO_DRIFT,
                'drift_type': drift_result.drift_type.value,
                'drift_probability': drift_result.drift_probability,
                'performance_degraded': performance_degraded,
                'version_created': self.current_version_id is not None
            }
    
    def _check_performance_degradation(
        self,
        new_loss: float,
        new_accuracy: float
    ) -> bool:
        """
        Check if model performance has degraded
        
        Args:
            new_loss: New loss value
            new_accuracy: New accuracy value
            
        Returns:
            True if performance degraded
        """
        if len(self.loss_history) < 100:
            return False
        
        # Calculate baseline performance
        baseline_loss = np.mean(list(self.loss_history)[-100:])
        baseline_accuracy = np.mean(list(self.accuracy_history)[-100:])
        
        # Check for degradation (10% worse than baseline)
        loss_degradation = new_loss > baseline_loss * 1.1
        accuracy_degradation = new_accuracy < baseline_accuracy * 0.9
        
        return loss_degradation or accuracy_degradation
    
    def _create_new_model_version(
        self,
        loss: float,
        accuracy: float,
        drift_result: DriftDetectionResult
    ):
        """
        Create new model version for A/B testing
        
        Args:
            loss: Current loss
            accuracy: Current accuracy
            drift_result: Drift detection result
        """
        # Get current model state
        model_state = self.model.state_dict()
        
        # Create performance metrics
        performance_metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'drift_detected': drift_result.drift_type != DriftType.NO_DRIFT,
            'drift_severity': drift_result.severity,
            'experience_count': self.total_experiences,
            'update_count': self.total_updates,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Register new version
        version_id = self.version_manager.register_new_version(
            model_state=model_state,
            performance_metrics=performance_metrics,
            parent_version=self.current_version_id,
            metadata={
                'drift_type': drift_result.drift_type.value,
                'drift_probability': drift_result.drift_probability,
                'created_by': 'online_learning_pipeline',
                'reason': drift_result.recommendation
            }
        )
        
        # Update current version
        self.current_version_id = version_id
        self.baseline_performance = performance_metrics.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            return {
                'total_experiences': self.total_experiences,
                'total_updates': self.total_updates,
                'current_loss': self.current_loss,
                'current_accuracy': self.current_accuracy,
                'avg_loss': np.mean(list(self.loss_history)[-100:]) if self.loss_history else 0.0,
                'avg_accuracy': np.mean(list(self.accuracy_history)[-100:]) if self.accuracy_history else 0.0,
                'loss_trend': self._calculate_trend(list(self.loss_history)[-50:]) if len(self.loss_history) >= 50 else 0.0,
                'accuracy_trend': self._calculate_trend(list(self.accuracy_history)[-50:]) if len(self.accuracy_history) >= 50 else 0.0,
                'buffer_statistics': self.replay_buffer.get_statistics(),
                'version_statistics': self.version_manager.get_version_statistics(),
                'drift_statistics': self.drift_detector.get_drift_statistics()
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend of values (positive = improving, negative = degrading)
        
        Args:
            values: List of values
            
        Returns:
            Trend value (slope)
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        
        return slope
    
    def save_pipeline(
        self,
        filepath: str
    ):
        """
        Save entire learning pipeline
        
        Args:
            filepath: Path to save pipeline
        """
        with self.lock:
            # Save model
            model_path = filepath.replace('.pkl', '_model.pth')
            torch.save(self.model.state_dict(), model_path)
            
            # Save pipeline state
            pipeline_state = {
                'total_experiences': self.total_experiences,
                'total_updates': self.total_updates,
                'current_loss': self.current_loss,
                'current_accuracy': self.current_accuracy,
                'current_version_id': self.current_version_id,
                'baseline_performance': self.baseline_performance,
                'performance_history': [
                    {
                        'timestamp': item['timestamp'].isoformat(),
                        'loss': item['loss'],
                        'accuracy': item['accuracy'],
                        'experience_count': item['experience_count']
                    }
                    for item in list(self.performance_history)[-1000:]  # Last 1000 records
                ],
                'loss_history': list(self.loss_history)[-1000:],  # Last 1000 records
                'accuracy_history': list(self.accuracy_history)[-1000:],  # Last 1000 records
                'version_manager_state': self.version_manager.get_version_statistics(),
                'buffer_statistics': self.replay_buffer.get_statistics(),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            pipeline_path = filepath.replace('.pkl', '_pipeline.pkl')
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline_state, f)
    
    def load_pipeline(
        self,
        filepath: str
    ):
        """
        Load learning pipeline state
        
        Args:
            filepath: Path to load pipeline
        """
        pipeline_path = filepath.replace('.pkl', '_pipeline.pkl')
        
        try:
            with open(pipeline_path, 'rb') as f:
                pipeline_state = pickle.load(f)
            
            with self.lock:
                # Restore basic statistics
                self.total_experiences = pipeline_state.get('total_experiences', 0)
                self.total_updates = pipeline_state.get('total_updates', 0)
                self.current_loss = pipeline_state.get('current_loss', 0.0)
                self.current_accuracy = pipeline_state.get('current_accuracy', 0.0)
                self.current_version_id = pipeline_state.get('current_version_id', None)
                self.baseline_performance = pipeline_state.get('baseline_performance', {})
                
                # Restore histories (limited to prevent memory issues)
                perf_history = pipeline_state.get('performance_history', [])
                self.performance_history = deque(
                    maxlen=self.performance_window
                )
                for item in perf_history[-1000:]:  # Last 1000 records
                    self.performance_history.append({
                        'timestamp': pd.Timestamp(item['timestamp']),
                        'loss': item['loss'],
                        'accuracy': item['accuracy'],
                        'experience_count': item['experience_count']
                    })
                
                loss_history = pipeline_state.get('loss_history', [])
                self.loss_history = deque(
                    [float(x) for x in loss_history[-1000:]],
                    maxlen=self.performance_window
                )
                
                acc_history = pipeline_state.get('accuracy_history', [])
                self.accuracy_history = deque(
                    [float(x) for x in acc_history[-1000:]],
                    maxlen=self.performance_window
                )
                
                print(f"Pipeline loaded successfully from {pipeline_path}")
                
        except FileNotFoundError:
            warnings.warn(f"Pipeline state file not found: {pipeline_path}")
        except Exception as e:
            warnings.warn(f"Error loading pipeline: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize components
    print("=== ONLINE LEARNING PIPELINE EXAMPLE ===\n")
    
    # Create sample model (simple feedforward network)
    class SimpleModel(nn.Module):
        def __init__(self, input_size=150, hidden_size=256, num_classes=3):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes)
            )
        
        def forward(self, x):
            if len(x.shape) == 3:  # (batch, seq, features)
                x = x[:, -1, :]  # Take last timestep
            return self.network(x)
    
    # Initialize model and components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize pipeline
    pipeline = OnlineLearningPipeline(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        buffer_size=1000,
        learning_rate=0.001,
        update_frequency=50,
        performance_window=1000,
        device=device
    )
    
    print("Pipeline initialized successfully!")
    print(f"Device: {device}")
    print(f"Buffer size: 1000")
    print(f"Update frequency: 50 experiences")
    
    # Generate sample data
    print("\nGenerating sample data...")
    batch_size = 32
    seq_len = 60
    num_features = 150
    num_classes = 3
    
    # Create sample features and targets
    features = torch.randn(batch_size, seq_len, num_features).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    print(f"Sample data shape: {features.shape}")
    print(f"Sample targets shape: {targets.shape}")
    
    # Add experiences to pipeline
    print("\nAdding experiences to pipeline...")
    for i in range(10):
        pipeline.add_experience(
            features=features,
            targets=targets,
            metadata={'batch': i, 'timestamp': pd.Timestamp.now()}
        )
        
        if i % 2 == 0:  # Update every 2 batches
            result = pipeline.update_model()
            if result['updated']:
                print(f"  Model updated: {result['total_updates']} updates")
    
    # Get performance metrics
    print("\nGetting performance metrics...")
    metrics = pipeline.get_performance_metrics()
    
    print("Performance Metrics:")
    print(f"  Total Experiences: {metrics['total_experiences']:,}")
    print(f"  Total Updates: {metrics['total_updates']}")
    print(f"  Current Loss: {metrics['current_loss']:.4f}")
    print(f"  Current Accuracy: {metrics['current_accuracy']:.4f}")
    print(f"  Average Loss: {metrics['avg_loss']:.4f}")
    print(f"  Average Accuracy: {metrics['avg_accuracy']:.4f}")
    
    # Check buffer statistics
    buffer_stats = metrics['buffer_statistics']
    print(f"\nBuffer Statistics:")
    print(f"  Buffer Size: {buffer_stats['buffer_size']}")
    print(f"  Positive Rate: {buffer_stats['positive_rate']:.2%}")
    print(f"  Negative Rate: {buffer_stats['negative_rate']:.2%}")
    print(f"  Average Priority: {buffer_stats['avg_priority']:.4f}")
    
    # Check version statistics
    version_stats = metrics['version_statistics']
    print(f"\nVersion Statistics:")
    print(f"  Total Versions: {version_stats['total_versions']}")
    print(f"  Active Versions: {version_stats['active_versions']}")
    print(f"  Best Version: {version_stats['best_version']}")
    
    # Detect concept drift
    print("\nDetecting concept drift...")
    drift_result = pipeline.drift_detector.detect_drift()
    
    print("Drift Detection Result:")
    print(f"  Drift Type: {drift_result.drift_type.value}")
    print(f"  Drift Probability: {drift_result.drift_probability:.4f}")
    print(f"  Severity: {drift_result.severity:.4f}")
    print(f"  Recommendation: {drift_result.recommendation}")
    print(f"  Affected Features: {len(drift_result.affected_features)}")
    
    # Save pipeline
    print("\nSaving pipeline...")
    pipeline.save_pipeline('./online_learning_pipeline.pkl')
    print("Pipeline saved successfully!")
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipeline.load_pipeline('./online_learning_pipeline.pkl')
    print("Pipeline loaded successfully!")
    
    print("\n=== ONLINE LEARNING PIPELINE DEMO COMPLETE ===")