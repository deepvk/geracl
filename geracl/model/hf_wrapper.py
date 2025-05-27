from transformers import PreTrainedModel

from geracl.model.config import GeraclConfigHF
from geracl.model.geracl_core import GeraclCore


class GeraclHF(PreTrainedModel):
    config_class = GeraclConfigHF

    def __init__(self, config: GeraclConfigHF):
        super().__init__(config)
        self._classification_core = GeraclCore(
            embedder_name=config.embedder_name,
            ffn_dim=config.ffn_dim,
            ffn_classes_dropout=config.ffn_classes_dropout,
            ffn_text_dropout=config.ffn_text_dropout,
            device=config.device,
            tokenizer_len=config.tokenizer_len,
            pooling_type=config.pooling_type,
            loss_args=config.loss_args,
        )
        self.post_init()

    # delegate to the core
    def forward(self, *args, **kwargs):
        return self._classification_core(*args, **kwargs)
