"""A TabPFN classifier that finetunes the underlying model for a single task.

This module provides the FinetunedTabPFNClassifier class, which wraps TabPFN
and allows fine-tuning on a specific dataset using the familiar scikit-learn
.fit() and .predict() API.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils.validation import check_is_fitted

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks,
    meta_dataset_collator,
)
from tabpfn.finetuning.finetuned_base import EvalResult, FinetunedTabPFNBase
from tabpfn.finetuning.train_util import clone_model_for_evaluation

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tabpfn.constants import XType, YType
    from tabpfn.finetuning.data_util import ClassifierBatch


def _compute_classification_loss(
    *,
    logits_BLQ: torch.Tensor,
    targets_BQ: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-entropy training loss.

    Shapes suffixes:
        B=batch * estimators, L=logits, Q=n_queries.

    Args:
        logits_BLQ: Raw logits of shape (B*E, L, Q).
        targets_BQ: Integer class targets of shape (B*E, Q).

    Returns:
        A scalar loss tensor.
    """
    return torch.nn.functional.cross_entropy(logits_BLQ, targets_BQ)


class FinetunedTabPFNClassifier(FinetunedTabPFNBase, ClassifierMixin):
    """A scikit-learn compatible wrapper for fine-tuning the TabPFNClassifier.

    This class encapsulates the fine-tuning loop, allowing you to fine-tune
    TabPFN on a specific dataset using the familiar .fit() and .predict() API.

    Args:
        device: The device to run the model on. Defaults to "cuda".
        epochs: The total number of passes through the fine-tuning data.
            Defaults to 30.
        time_limit: Time limit in seconds for fine-tuning.
            If None, no time limit is applied. Defaults to None.
        learning_rate: The learning rate for the AdamW optimizer. A small value
            is crucial for stable fine-tuning. Defaults to 1e-5.
        weight_decay: The weight decay for the AdamW optimizer. Defaults to 0.01.
        validation_split_ratio: Fraction of the original training data reserved
            as a validation set for early stopping and monitoring. Defaults to 0.1.
        n_finetune_ctx_plus_query_samples: The total number of samples per
            meta-dataset during fine-tuning (context plus query) before applying
            the `finetune_ctx_query_split_ratio`. Defaults to 10_000.
        finetune_ctx_query_split_ratio: The proportion of each fine-tuning
            meta-dataset to use as query samples for calculating the loss. The
            remainder is used as context. Defaults to 0.2.
        n_inference_subsample_samples: The total number of subsampled training
            samples per estimator during validation and final inference.
            Defaults to 50_000.
        random_state: Seed for reproducibility of data splitting and model
            initialization. Defaults to 0.
        early_stopping: Whether to use early stopping based on validation
            performance. Defaults to True.
        early_stopping_patience: Number of epochs to wait for improvement before
            early stopping. Defaults to 8.
        min_delta: Minimum change in metric to be considered as an improvement.
            Defaults to 1e-4.
        grad_clip_value: Maximum norm for gradient clipping. If None, gradient
            clipping is disabled. Gradient clipping helps stabilize training by
            preventing exploding gradients. Defaults to 1.0.
        use_lr_scheduler: Whether to use a learning rate scheduler (linear warmup
            with optional cosine decay) during fine-tuning. Defaults to True.
        lr_warmup_only: If True, only performs linear warmup to the base learning
            rate and then keeps it constant. If False, applies cosine decay after
            warmup. Defaults to False.
        n_estimators_finetune: If set, overrides `n_estimators` of the underlying
            estimator only during fine-tuning to control the number of
            estimators (ensemble size) used in the training loop. If None, the
            value from `kwargs` or the estimator default is used.
            Defaults to 2.
        n_estimators_validation: If set, overrides `n_estimators` only for
            validation-time evaluation during fine-tuning (early-stopping /
            monitoring). If None, the value from `kwargs` or the
            estimator default is used. Defaults to 2.
        n_estimators_final_inference: If set, overrides `n_estimators` only for
            the final fitted inference model that is used after fine-tuning. If
            None, the value from `kwargs` or the estimator default is used.
            Defaults to 8.
        use_activation_checkpointing: Whether to use activation checkpointing to
            reduce memory usage. Defaults to True.
        save_checkpoint_interval: Number of epochs between checkpoint saves. This
            only has an effect if `output_dir` is provided during the `fit()` call.
            If None, no intermediate checkpoints are saved. The best model checkpoint
            is always saved regardless of this setting. Defaults to 10.

        FinetunedTabPFNClassifier specific arguments:

        extra_classifier_kwargs: Additional keyword arguments to pass to the
            underlying `TabPFNClassifier`, such as `n_estimators`.
        eval_metric: The primary metric to monitor during fine-tuning.
            For classification, this is ROC AUC by default.
            The choices are: "roc_auc", "log_loss"
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        device: str = "cuda",
        epochs: int = 30,
        time_limit: int | None = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        validation_split_ratio: float = 0.1,
        n_finetune_ctx_plus_query_samples: int = 10_000,
        finetune_ctx_query_split_ratio: float = 0.2,
        n_inference_subsample_samples: int = 50_000,
        random_state: int = 0,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        use_lr_scheduler: bool = True,
        lr_warmup_only: bool = False,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 8,
        use_activation_checkpointing: bool = True,
        save_checkpoint_interval: int | None = 10,
        extra_classifier_kwargs: dict[str, Any] | None = None,
        eval_metric: Literal["roc_auc", "log_loss"] | None = None,
        image_embedding_dim: int | None = None,
        image_mlp_ratio: int = 2,
    ):
        super().__init__(
            device=device,
            epochs=epochs,
            time_limit=time_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split_ratio=validation_split_ratio,
            n_finetune_ctx_plus_query_samples=n_finetune_ctx_plus_query_samples,
            finetune_ctx_query_split_ratio=finetune_ctx_query_split_ratio,
            n_inference_subsample_samples=n_inference_subsample_samples,
            random_state=random_state,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            grad_clip_value=grad_clip_value,
            use_lr_scheduler=use_lr_scheduler,
            lr_warmup_only=lr_warmup_only,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            use_activation_checkpointing=use_activation_checkpointing,
            save_checkpoint_interval=save_checkpoint_interval,
        )
        self.extra_classifier_kwargs = extra_classifier_kwargs
        self.eval_metric = eval_metric
        self.image_embedding_dim = image_embedding_dim
        self.image_mlp_ratio = image_mlp_ratio

    @property
    @override
    def _estimator_kwargs(self) -> dict[str, Any]:
        """Return the classifier-specific kwargs."""
        return self.extra_classifier_kwargs or {}

    @property
    @override
    def _model_type(self) -> Literal["classifier", "regressor"]:
        """Return the model type string."""
        return "classifier"

    @property
    @override
    def _metric_name(self) -> str:
        """Return the name of the primary metric."""
        return "ROC AUC"

    @override
    def _create_estimator(self, config: dict[str, Any]) -> TabPFNClassifier:
        """Create the TabPFNClassifier with the given config."""
        return TabPFNClassifier(
            **config,
            fit_mode="batched",
            differentiable_input=False,
        )

    @override
    def _setup_estimator(self) -> None:
        """Set up softmax temperature after estimator creation."""
        self.finetuned_estimator_.softmax_temperature_ = (
            self.finetuned_estimator_.softmax_temperature
        )
        if self.image_embedding_dim is not None:
            hidden_dim = self.image_embedding_dim * self.image_mlp_ratio
            self.image_projector_ = nn.Sequential(
                nn.Linear(self.image_embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.image_embedding_dim),
            ).to(self.device)

    @override
    def _on_model_initialized(self) -> None:
        """Attach optional image projector to the initialized model."""
        if hasattr(self, "image_projector_"):
            self.finetuned_estimator_.model_.add_module(
                "image_projector", self.image_projector_
            )
            if not hasattr(self, "_n_classes_train"):
                raise RuntimeError("n_classes must be known before model initialization.")
            tab_feature_dim = self.finetuned_estimator_.model_.ninp
            multimodal_feature_dim = tab_feature_dim + self.image_embedding_dim
            self.multimodal_classifier_ = nn.Linear(
                multimodal_feature_dim,
                self._n_classes_train,
            ).to(self.device)
            self.finetuned_estimator_.model_.add_module(
                "multimodal_classifier", self.multimodal_classifier_
            )

    @override
    def _setup_batch(self, batch: ClassifierBatch) -> None:  # type: ignore[override]
        """Prepare optional image modality for multimodal fusion."""
        if not hasattr(self, "image_projector_"):
            return
        if batch.X_image_context is None or batch.X_image_query is None:
            raise ValueError("X_image must be provided when image_embedding_dim is set.")

    def _encode_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Apply transformer-like FFN block with residual connection."""
        projected = self.image_projector_(image_features)
        return projected + image_features

    def _extract_tabpfn_raw_outputs(
        self, X_query_batch: list[torch.Tensor] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw logits and pre-classifier tab features for each estimator."""
        outputs = []
        embeddings = []
        for output, config in self.finetuned_estimator_.executor_.iter_outputs(
            X_query_batch,
            autocast=self.finetuned_estimator_.use_autocast_,
            only_return_standard_out=False,
        ):
            if not isinstance(output, dict):
                raise RuntimeError("Expected dict output with embeddings.")
            logits = output["standard"]
            test_embeddings = output["test_embeddings"]

            if logits.ndim == 2:
                logits = logits.unsqueeze(1)
                test_embeddings = test_embeddings.unsqueeze(1)
                config_list = [config]
            elif logits.ndim == 3:
                config_list = config
            else:
                raise ValueError(f"Output tensor must be 2d or 3d, got {logits.ndim}d")

            logits_batch = []
            embedding_batch = []
            for i, batch_config in enumerate(config_list):
                if batch_config.class_permutation is None:
                    logits_batch.append(
                        logits[:, i, : self.finetuned_estimator_.n_classes_]
                    )
                else:
                    if len(batch_config.class_permutation) != self.finetuned_estimator_.n_classes_:
                        use_perm = np.arange(self.finetuned_estimator_.n_classes_)
                        use_perm[: len(batch_config.class_permutation)] = (
                            batch_config.class_permutation
                        )
                    else:
                        use_perm = batch_config.class_permutation
                    logits_batch.append(logits[:, i, use_perm])
                embedding_batch.append(test_embeddings[:, i, :])

            outputs.append(torch.stack(logits_batch, dim=1))
            embeddings.append(torch.stack(embedding_batch, dim=1))

        stacked_logits = torch.stack(outputs).transpose(0, 1).transpose(1, 2)
        stacked_embeddings = torch.stack(embeddings).transpose(0, 1).transpose(1, 2)
        return stacked_logits, stacked_embeddings

    def _fuse_multimodal_logits(
        self,
        tab_embeddings_QBED: torch.Tensor,
        image_query_BQD: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse tab and image features and produce logits."""
        if not hasattr(self, "multimodal_classifier_"):
            raise RuntimeError("multimodal_classifier_ must be initialized.")
        image_query = self._encode_image_features(image_query_BQD.to(self.device))
        image_query = image_query.permute(1, 0, 2)
        Q, B, E, _ = tab_embeddings_QBED.shape
        image_query = image_query[:, :, None, :].expand(Q, B, E, -1)
        fused_features = torch.cat([tab_embeddings_QBED, image_query], dim=-1)
        return self.multimodal_classifier_(fused_features)

    @override
    def _should_skip_batch(self, batch: ClassifierBatch) -> bool:  # type: ignore[override]
        """Check if the batch should be skipped."""
        ctx_unique = torch.unique(
            torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_context])
        )
        qry_unique = torch.unique(
            torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_query])
        )

        query_in_context = torch.isin(qry_unique, ctx_unique, assume_unique=True)
        if not bool(query_in_context.all()):
            missing_labels = qry_unique[~query_in_context].detach().cpu().numpy()
            context_labels = ctx_unique.detach().cpu().numpy()
            logger.warning(
                "Skipping batch: query labels %s are not a subset of context labels %s",
                missing_labels,
                context_labels,
            )
            return True
        return False

    @override
    def _forward_with_loss(self, batch: ClassifierBatch) -> torch.Tensor:  # type: ignore[override]
        """Perform forward pass and compute and return cross-entropy loss.

        Args:
            batch: The ClassifierBatch containing preprocessed context and
                query data.

        Returns:
            The computed cross-entropy loss tensor.
        """
        X_query_batch = batch.X_query
        y_query_batch = batch.y_query

        # shape suffix: Q=n_queries, B=batch(=1), E=estimators, L=logits
        logits_QBEL = self.finetuned_estimator_.forward(
            X_query_batch,
            return_raw_logits=True,
        )
        if hasattr(self, "image_projector_"):
            if batch.X_image_query is None:
                raise ValueError(
                    "X_image_query must be provided when image_embedding_dim is set."
                )
            _, tab_embeddings_QBED = self._extract_tabpfn_raw_outputs(X_query_batch)
            logits_QBEL = self._fuse_multimodal_logits(
                tab_embeddings_QBED,
                batch.X_image_query,
            )

        Q, B, E, L = logits_QBEL.shape
        assert y_query_batch.shape[1] == Q
        assert B == 1
        assert self.n_estimators_finetune == E
        assert self.finetuned_estimator_.n_classes_ == L

        # Reshape for CE loss: treat estimator dim as batch dim
        # permute to shape (B, E, L, Q) then reshape to (B*E, L, Q)
        logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
        targets_BQ = y_query_batch.repeat(B * self.n_estimators_finetune, 1).to(
            self.device
        )

        return _compute_classification_loss(
            logits_BLQ=logits_BLQ,
            targets_BQ=targets_BQ,
        )

    @override
    def _evaluate_model(
        self,
        eval_config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> EvalResult:
        """Evaluate the classifier using ROC AUC and log loss."""
        eval_classifier = clone_model_for_evaluation(
            self.finetuned_estimator_,
            eval_config,
            TabPFNClassifier,
        )
        eval_classifier.fit(X_train, y_train)

        try:
            probabilities = eval_classifier.predict_proba(X_val)  # type: ignore
            if probabilities.shape[1] > 2:
                roc_auc = roc_auc_score(y_val, probabilities, multi_class="ovr")
            else:
                roc_auc = roc_auc_score(y_val, probabilities[:, 1])
            log_loss_score = log_loss(y_val, probabilities)
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"An error occurred during evaluation: {e}")
            roc_auc, log_loss_score = np.nan, np.nan

        if self.eval_metric == "roc_auc":
            primary_metric = roc_auc
        elif self.eval_metric == "log_loss":
            primary_metric = -log_loss_score
        else:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric}")

        return EvalResult(
            primary=primary_metric,  # pyright: ignore[reportArgumentType]
            secondary={"log_loss": log_loss_score, "roc_auc": roc_auc},
        )

    @override
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current ROC AUC is better (higher) than best."""
        return current > best + self.min_delta

    @override
    def _get_initial_best_metric(self) -> float:
        """Return -inf for maximization."""
        return -np.inf

    @override
    def _get_checkpoint_metrics(self, eval_result: EvalResult) -> dict[str, float]:
        """Return metrics for checkpoint saving."""
        return {
            "primary_metric": eval_result.primary,
            "log_loss": eval_result.secondary.get("log_loss", np.nan),
            "roc_auc": eval_result.secondary.get("roc_auc", np.nan),
        }

    @override
    def _get_valid_finetuning_query_size(
        self, *, query_size: int, y_train: np.ndarray
    ) -> int:
        """Calculate a valid finetuning query size."""
        # finetuning_query_size should be greater or equal to the number of classes
        assert y_train is not None, (
            "y_train required to compute finetuning query size for classification."
        )
        n_classes = len(np.unique(y_train))
        return max(query_size, n_classes)

    @override
    def _log_epoch_evaluation(
        self, epoch: int, eval_result: EvalResult, mean_train_loss: float | None
    ) -> None:
        """Log evaluation results for classification."""
        mean_train_loss = "N/A" if mean_train_loss is None else f"{mean_train_loss:.4f}"
        metric = eval_result.primary
        logger.info(
            f"📊 Epoch {epoch + 1} Evaluation | Val {self.eval_metric}: {metric:.4f}"
            f"\n\t Train Loss: {mean_train_loss}"
            f"\n\t Secondary Metrics: {eval_result.secondary}",
        )

    @override
    def _setup_inference_model(
        self, final_inference_eval_config: dict[str, Any]
    ) -> None:
        """Set up the final inference classifier."""
        finetuned_inference_classifier = clone_model_for_evaluation(
            self.finetuned_estimator_,
            final_inference_eval_config,
            TabPFNClassifier,
        )
        self.finetuned_inference_classifier_ = finetuned_inference_classifier
        self.finetuned_inference_classifier_.fit_mode = "fit_preprocessors"  # type: ignore
        self.finetuned_inference_classifier_.fit(self.X_, self.y_)  # type: ignore

    @override
    def fit(
        self,
        X: XType,
        y: YType | XType | None = None,
        X_image: XType | YType | None = None,
        X_val: XType | None = None,
        y_val: YType | None = None,
        output_dir: Path | None = None,
    ) -> FinetunedTabPFNClassifier:
        """Fine-tune the TabPFN model on the provided training data.

        Args:
            X: The training input samples of shape (n_samples, n_features).
            y: The target values of shape (n_samples,).
            X_val: Optional validation input samples.
            y_val: Optional validation target values.
            output_dir: Directory path for saving checkpoints. If None, no
                checkpointing is performed and progress will be lost if
                training is interrupted.

        Returns:
            The fitted instance itself.
        """
        if self.eval_metric is None:
            self.eval_metric = "roc_auc"

        # Backward-compatible path: historical API allowed fit(X, X_image, y).
        if y is not None and X_image is not None:
            y_arr = np.asarray(y)
            x_img_arr = np.asarray(X_image)
            y_looks_like_features = y_arr.ndim > 1 and y_arr.shape[1] > 1
            x_img_looks_like_labels = x_img_arr.ndim == 1 or (
                x_img_arr.ndim == 2 and x_img_arr.shape[1] == 1
            )
            if y_looks_like_features and x_img_looks_like_labels:
                warnings.warn(
                    "Detected deprecated call pattern fit(X, X_image, y). "
                    "Please use fit(X, y, X_image=...) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                y, X_image = X_image, y

        if y is None:
            raise ValueError("Missing labels y. Use fit(X, y, ...) to train the model.")

        if self.image_embedding_dim is not None and X_image is None:
            raise ValueError(
                "X_image must be provided when image_embedding_dim is set. "
                "Use fit(X, y, X_image=...)."
            )

        if X_image is not None:
            if len(X_image) != len(X):
                raise ValueError("X and X_image must have the same number of rows.")
            if (
                self.image_embedding_dim is not None
                and np.asarray(X_image).shape[1] != self.image_embedding_dim
            ):
                raise ValueError(
                    "X_image feature dimension does not match image_embedding_dim."
                )

        self._n_classes_train = len(np.unique(np.asarray(y)))

        self.X_image_ = X_image

        super().fit(
            X,
            y,  # type: ignore[arg-type]
            X_image=X_image if y is not None else None,
            X_val=X_val,
            y_val=y_val,
            output_dir=output_dir,
        )
        return self

    def _predict_proba_with_image(self, X: XType, X_image: XType) -> np.ndarray:
        """Predict probabilities with the optional image modality enabled."""
        if not hasattr(self, "image_projector_"):
            raise ValueError(
                "X_image was provided but image modality is not enabled. "
                "Set image_embedding_dim when constructing the classifier."
            )

        check_is_fitted(self)
        if self.X_image_ is None:
            raise ValueError("Image modality was not provided during fit().")

        X_train = np.asarray(self.X_)
        y_train = np.asarray(self.y_)
        X_query = np.asarray(X)
        X_image_train = np.asarray(self.X_image_)
        X_image_query = np.asarray(X_image)

        if len(X_query) != len(X_image_query):
            raise ValueError(
                "X and X_image must have the same number of rows during predict."
            )

        n_train = len(X_train)
        X_full = np.concatenate([X_train, X_query], axis=0)
        y_full = np.concatenate([y_train, np.zeros(len(X_query), dtype=y_train.dtype)])
        X_image_full = np.concatenate([X_image_train, X_image_query], axis=0)

        def _split_train_query(x_arr, y_arr, *, stratify=None):
            del stratify
            return (
                x_arr[:n_train],
                x_arr[n_train:],
                y_arr[:n_train],
                y_arr[n_train:],
            )

        inference_dataset = get_preprocessed_dataset_chunks(
            calling_instance=self.finetuned_estimator_,
            X_raw=X_full,
            y_raw=y_full,
            X_image_raw=X_image_full,
            split_fn=_split_train_query,
            max_data_size=None,
            model_type=self._model_type,
            equal_split_size=False,
            seed=self.random_state,
            shuffle=False,
            force_no_stratify=True,
        )
        inference_batch = meta_dataset_collator([inference_dataset[0]])
        self._setup_batch(inference_batch)

        self.finetuned_estimator_.fit_from_preprocessed(
            inference_batch.X_context,
            inference_batch.y_context,
            inference_batch.cat_indices,
            inference_batch.configs,
        )
        raw_logits = self.finetuned_estimator_.forward(
            inference_batch.X_query,
            return_raw_logits=True,
        )
        if hasattr(self, "image_projector_"):
            if inference_batch.X_image_query is None:
                raise ValueError(
                    "X_image_query must be provided when image modality is enabled."
                )
            _, tab_embeddings = self._extract_tabpfn_raw_outputs(inference_batch.X_query)
            raw_logits = self._fuse_multimodal_logits(
                tab_embeddings,
                inference_batch.X_image_query,
            )
        # raw_logits shape: (N, B, E, C) with B=1. Convert to (E, N, C).
        raw_logits = raw_logits[:, 0].permute(1, 0, 2)
        probas = self.finetuned_estimator_.logits_to_probabilities(raw_logits)
        return probas.float().detach().cpu().numpy()

    def predict_proba(
        self,
        X: XType,
        X_image: XType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict class probabilities for X.

        Args:
            X: The input samples of shape (n_samples, n_features).
            **kwargs: Additional keyword arguments to pass to the underlying
                inference classifier.

        Returns:
            The class probabilities of the input samples with shape
            (n_samples, n_classes).
        """
        check_is_fitted(self)

        if X_image is not None:
            return self._predict_proba_with_image(X, X_image)

        return self.finetuned_inference_classifier_.predict_proba(X, **kwargs)  # type: ignore

    @override
    def predict(
        self,
        X: XType,
        X_image: XType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict the class for X.

        Args:
            X: The input samples of shape (n_samples, n_features).
            **kwargs: Additional keyword arguments to pass to the underlying
                inference classifier.

        Returns:
            The predicted classes with shape (n_samples,).
        """
        check_is_fitted(self)

        if X_image is not None:
            probas = self.predict_proba(X, X_image=X_image, **kwargs)
            return np.argmax(probas, axis=1)

        return self.finetuned_inference_classifier_.predict(X, **kwargs)  # type: ignore
