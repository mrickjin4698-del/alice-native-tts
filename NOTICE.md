# Third-party notices

The Mandarin pronunciation tables, BERT tokenizer data, and pinyin splitting
code in `src/frontend/` are adapted from SmileSky's GPT-SoVITS Rust project:
<https://github.com/mzdk100/GPT-SoVITS>, commit
`39dcc7f8171bd7c909058486c85ccd5580e085a9` (Apache-2.0). The upstream license
text is included in `LICENSE`.

The v2Pro graph interface, English GRU inference, text segmentation, BERT
alignment, reference resampling, and cancellation code were implemented for
Project Alice.

The phoneme ordering in the native pack tool follows the g2p_en checkpoint
format from Kyubyong Park and Jongseok Kim's Apache-2.0 g2p project.
[g2p_en](https://github.com/Kyubyong/g2p)

The weight-free v2Pro operator graphs in `src/checkpoint_native/graphs/` are
built from the Apache-2.0 GPT-SoVITS minimal-inference architecture at
`6916eb2e9c227aeb28468ce6fa86ad5b17ead532`. The maintainer script
`tools/generate-v2pro-checkpoint-graphs.py` rebuilds them with PyTorch 2.7 and
ONNX opset 20 using freshly initialized modules. The FP16 templates use ONNX
Runtime's float16 converter on the weight-free graphs. Inference binds
checkpoint tensors to the external inputs in memory. [GPT-SoVITS minimal inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference/tree/6916eb2e9c227aeb28468ce6fa86ad5b17ead532)

Model packs are separate from this source distribution. Each pack must retain
the applicable provenance and license terms for GPT-SoVITS, G2PW, Chinese
RoBERTa, HuBERT, ERes2Net, g2p_en, CMUdict, and any custom voice weights or
reference recordings it contains.
