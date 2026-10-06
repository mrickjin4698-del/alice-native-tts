# Third-party notices

The Mandarin pronunciation tables, BERT tokenizer data, and pinyin splitting
code in `src/frontend/` are adapted from SmileSky's GPT-SoVITS Rust project:
<https://github.com/mzdk100/GPT-SoVITS>, commit
`39dcc7f8171bd7c909058486c85ccd5580e085a9` (Apache-2.0). The upstream license
text is included in `LICENSE`.

The v2Pro graph interface, English GRU inference, text segmentation, BERT
alignment, reference resampling, and cancellation code were implemented for
Project Alice.

Model packs are separate from this source distribution. Each pack must retain
the applicable provenance and license terms for GPT-SoVITS, G2PW, Chinese
RoBERTa, HuBERT, ERes2Net, g2p_en, CMUdict, and any custom voice weights or
reference recordings it contains.
