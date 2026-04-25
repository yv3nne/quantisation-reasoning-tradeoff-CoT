import logging, os
from dataclasses import dataclass, field

import torch, transformers
from gptqmodel import GPTQModel, QuantizeConfig


@dataclass
class QuantConf:
    model_name: str
    quant_bits: int
    quant_groupsize: int
    handle: str
    calibration_data: list
    base_path: str
    path: str = field(init=False, repr=True)

    def __post_init__(self):
        self.path = os.path.join(
            self.base_path,
            f"{self.model_name}_Q_{self.quant_bits}B{self.quant_groupsize}G",
        )
        os.makedirs(self.base_path, exist_ok=True)

    def quantise(self):
        if os.path.exists(self.path) and os.path.exists(os.path.join(self.path, "config.json")):
            logging.info(f"[QUANTISED] Found existing model at {self.path} — skipping")
            return

        quant_config = QuantizeConfig(
            bits=self.quant_bits,
            group_size=self.quant_groupsize,
            desc_act=False,
        )
        logging.info(f"[QUANTISING] {self.handle} → {self.quant_bits}b / {self.quant_groupsize}g")

        tok = transformers.AutoTokenizer.from_pretrained(self.handle)
        tok.pad_token = tok.eos_token

        try:
            tmp_model = GPTQModel.from_pretrained(
                self.handle,
                quant_config,
                torch_dtype=torch.float16,  # V100 has no bf16
            )
        except (TypeError, ValueError) as e:
            msg = str(e)
            if "isn't supported yet" in msg or "does not recognize this architecture" in msg:
                logging.warning(f"[SKIP] {self.handle}: not supported by this environment ({e}) — skipping")
                return
            raise

        # gptqmodel 1.4.0 expects list of dicts {"input_ids": [int, ...]}
        logging.info(f"[QUANTISING] Tokenizing {len(self.calibration_data)} calibration samples")
        tokenized_cal = [
            tok(text, truncation=True, max_length=512)
            for text in self.calibration_data
        ]

        tmp_model.quantize(tokenized_cal)
        os.makedirs(self.path, exist_ok=True)
        tmp_model.save(self.path)
        logging.info(f"[QUANTISING] Saved weights to {self.path}")
        del tmp_model
        torch.cuda.empty_cache()

        tok.save_pretrained(self.path)
        logging.info(f"[TOKENIZER] Saved to {self.path}")
