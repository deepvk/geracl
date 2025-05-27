from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class GeraclConfigHF(PretrainedConfig):
    model_type = "GeRaCl"
    is_composition = True

    def __init__(
        self,
        embedder_config=None,
        embedder_name=None,
        ffn_dim=None,
        ffn_classes_dropout=0.4,
        ffn_text_dropout=0.4,
        device="cuda",
        tokenizer_len=None,
        pooling_type="mean",
        loss_args={"loss_type": "bce"},
        **kwargs,
    ):
        if isinstance(embedder_config, dict):
            embedder_config["model_type"] = (
                embedder_config["model_type"] if "model_type" in embedder_config else "modernbert"
            )
            embedder_config = CONFIG_MAPPING[embedder_config["model_type"]](**embedder_config)
        elif embedder_config is None:
            embedder_config = CONFIG_MAPPING["modernbert"]()

        self.embedder_config = embedder_config
        self.embedder_name = embedder_name

        self.hidden_size = self.embedder_config.hidden_size

        if tokenizer_len is not None:
            self.tokenizer_len = tokenizer_len
        else:
            self.tokenizer_len = self.embedder_config.vocab_size

        if ffn_dim is None:
            self.ffn_dim = self.hidden_size * 2
        else:
            self.ffn_dim = ffn_dim

        self.ffn_classes_dropout = ffn_classes_dropout
        self.ffn_text_dropout = ffn_text_dropout

        self.pooling_type = pooling_type
        self.device = device
        self.loss_args = loss_args
        super().__init__(**kwargs)
