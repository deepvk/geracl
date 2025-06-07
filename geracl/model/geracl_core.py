import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, PretrainedConfig


class GeraclCore(nn.Module):
    """Core torch module that encapsulate all routines for zero-shot text classification.

    May be used as regular Torch module on inference: forward pass returns probabilities of choosing each possible class in a sample.

    Use HuggingFace models as backbone to embed tokens, e.g. "USER2-base":
    https://huggingface.co/deepvk/USER2-base
    """

    def __init__(
        self,
        embedder_name: str = "deepvk/USER-base",
        *,
        ffn_dim: int = 2048,
        ffn_classes_dropout: float = 0.1,
        ffn_text_dropout: float = 0.1,
        device: str = "cuda",
        tokenizer_len: int,
        pooling_type: str = "mean",
        loss_args: dict = None,
        embedder_config: PretrainedConfig | None = None,
    ):
        """
        :param embedder_name: name of pretrained HuggingFace model to embed tokens.
        :param ffn_dim: hidden dimension of mlp layers.
        :param ffn_classes_dropout: dropout of the mlp layer used for transforming input classes embeddings.
        :param ffn_text_dropout: dropout of the mlp layer used for transforming input text embedding.
        :param device: name of device to train the model on.
        :param tokenizer_len: Number of tokens in tokenizer.
        :param loss_args: Dict with arguments to choose appropriate loss function.
        """
        super().__init__()
        if pooling_type not in {"mean", "first"}:
            raise ValueError("Invalid pooling type config parameter.")
        self._pooling_type = pooling_type

        self._device = device
        if embedder_config is not None:
            self._token_embedder = AutoModel.from_config(embedder_config)
        else:
            self._token_embedder = AutoModel.from_pretrained(embedder_name).to(self._device)
            self._token_embedder.resize_token_embeddings(tokenizer_len)

        self._mlp_classes = nn.Sequential(
            nn.Linear(self._token_embedder.config.hidden_size, ffn_dim),
            nn.Dropout(ffn_classes_dropout),
            nn.GELU(),
            nn.Linear(ffn_dim, self._token_embedder.config.hidden_size),
        )

        self._mlp_text = nn.Sequential(
            nn.Linear(self._token_embedder.config.hidden_size, ffn_dim),
            nn.Dropout(ffn_text_dropout),
            nn.GELU(),
            nn.Linear(ffn_dim, self._token_embedder.config.hidden_size),
        )

    def _get_text_embeddings(
        self,
        embeddings: torch.Tensor,
        classes_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate token embeddings that correspond to the *input text* (mask value **-3**)
        and project the pooled vector through the text-MLP head.

        Args:
            embeddings (Tensor): shape **(batch_size, seq_len, hidden_dim)**
                Raw contextual embeddings produced by the encoder.

            classes_mask (Tensor): shape **(batch_size, seq_len)**
                Integer mask that tags every token:
                * **-3** -- genuine input-text tokens (these are the ones we pool)
                * **-5** -- extra prompt / scenario tokens (ignored here)
                * **-1/-2 / 0…N** -- other task-specific labels (ignored here)

        Returns:
            Tensor: shape **(batch_size, text_proj_dim)**
                One embedding per sample, summarising the entire input text after
                the MLP projection.
        """
        mask = classes_mask == -3

        text_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        text_embeddings = torch.sum(embeddings * text_mask_expanded, 1) / torch.clamp(
            text_mask_expanded.sum(1), min=1e-9
        )
        text_embeddings = self._mlp_text(text_embeddings)
        return text_embeddings

    def _get_classes_embeddings(
        self, embeddings: torch.Tensor, classes_mask: torch.Tensor, classes_count: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Build a single stack of per-class embeddings for the *entire* batch and project
        them through the class-MLP head.

        The exact pooling strategy depends on `self._pooling_type`:
        * **"mean"**  - mean of token vectors spanning each label span
        * **"first"** - first occurrence of each label (HF-style)

        Args:
            embeddings (Tensor): shape **(batch_size, seq_len, hidden_dim)**
                Contextual encoder output.

            classes_mask (Tensor): shape **(batch_size, seq_len)**
                Integer mask assigning every token to either
                * a class ID **0 … C-1**,
                * the two special “separator” labels **-1** and **-2** (used to distinguish
                positive and negative classes and set boundaries between classes),
                * or any negative value reserved for non-class tokens (e.g. **-3**, **-5**).

            classes_count (Tensor): shape **(batch_size,)**
                How many distinct classes appear in each sample.
                The sum of this vector equals the first dimension of the returned tensor.

        Returns:
            Tensor: shape **(total_classes_in_batch, class_proj_dim)**
                One embedding per class instance across the batch after
                the MLP projection, ordered batch-major (all classes of sample 0,
                then sample 1, …).
        """
        bs, _ = classes_mask.shape

        mlp_input = torch.empty(
            (classes_count.sum(), embeddings.shape[-1]), dtype=torch.float, device=embeddings.device
        )
        idx = 0
        for i in range(bs):
            sample_classes_mask = classes_mask[i]
            emb = embeddings[i]
            if self._pooling_type == "mean":
                sample_class_embeddings = []
                for label in range(classes_count[i]):
                    positions = torch.nonzero(sample_classes_mask == label, as_tuple=True)[0]
                    start_idx = positions.min()
                    end_idx = positions.max()

                    # +1 because we should include token on end_idx position
                    class_mean = emb[start_idx : end_idx + 1].mean(dim=0)
                    sample_class_embeddings.append(class_mean)
                sample_class_embeddings = torch.stack(sample_class_embeddings, dim=0)
            elif self._pooling_type == "first":
                positions_1 = torch.nonzero(sample_classes_mask == -1).squeeze(1)
                positions_2 = torch.nonzero(sample_classes_mask == -2).squeeze(1)
                sorted_classes_positions = torch.sort(torch.cat((positions_2, positions_1)))[0]
                sample_class_embeddings = emb[sorted_classes_positions]

            mlp_input[idx : (idx + classes_count[i])] = sample_class_embeddings
            idx = idx + classes_count[i]

        classes_embeddings = self._mlp_classes(mlp_input.to(self._device))
        return classes_embeddings

    def forward(self, input_ids: Tensor, attention_mask: Tensor, classes_mask: Tensor, classes_count: Tensor) -> Tensor:  # type: ignore
        """
        End-to-end zero-shot classification pass:
        embed the input text, embed each candidate class, and return a similarity
        score for every *text ↔ class* pair in the batch.

        Args:
            input_ids (Tensor): shape **(batch_size, seq_len)**
                Pre-tokenised input IDs.

            attention_mask (Tensor): shape **(batch_size, seq_len)**
                Standard HF mask — **1** for real tokens, **0** for padding.

            classes_mask (Tensor): shape **(batch_size, seq_len)**
                Per-token label map:
                * **0 … C-1** – tokens belonging to that class span
                * **-1**, **-2** – class separators (used to distinguish
                positive and negative classes and set boundaries between classes)
                * **-3** – tokens of the actual input text (pooled by
                `_get_text_embeddings`)
                * **-5** – prompt / scenario tokens (ignored here)

            classes_count (Tensor): shape **(batch_size,)**
                Number of distinct candidate classes in each sample.

        Returns:
            Tensor: shape **(batch_total_classes,)** where
            `batch_total_classes = classes_count.sum()`
            Dot-product similarities between the pooled text embedding
            of each sample and each of its class embeddings, ordered
            batch-major (all classes of sample 0, then sample 1, …).
        """
        embeddings = self._token_embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_embeddings = self._get_text_embeddings(embeddings, classes_mask)
        classes_embeddings = self._get_classes_embeddings(embeddings, classes_mask, classes_count)

        # [all_classes_in_batch, embedding_dim]
        wide_text_embeddings = torch.repeat_interleave(text_embeddings, classes_count, dim=0)

        similarities = torch.sum(classes_embeddings * wide_text_embeddings, dim=-1)

        return similarities
