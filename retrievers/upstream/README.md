# Fixed upstream code

These files are unmodified upstream sources, with their MIT licenses retained.

- `kasla_utils.py`: `DEEP-PolyU/KaSLA`, commit
  `f4dc879e0d4c1a47d0bef01af2616a84424abeb9`, `KaSLA_utils.py`.
- `linkalign_prompts.py`: `Satissss/LinkAlign`, commit
  `c036f38e0930843de9f000bf69b0bf79df79733a`,
  `prompts/PipelinePromptStore.py`.

The adapters are `retrievers/kasla_retriever.py` and
`retrievers/linkalign_official_retriever.py`. They do not import upstream SQL
generation/evaluation code. The KaSLA adapter's binary-linker input prefix is
from `src/utils_data_processing/prepare_inputs.py`,
`prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike`.
