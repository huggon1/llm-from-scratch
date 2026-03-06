### 从头训练一个 tokenizer

tokenizer 的作用是将字符序列转为数字序列，tokenizer 对应不同的粒度也有不同的分词方式。常见的一般有

- word base: 按照词进行分词
- character base：按照单字符进行分词
- subword tokenization：按照词的 subword 进行分词

其中 subword 粒度用的比较多，subword tokenizer 算法的核心思路就是尽可能的合并出现频次多的词。常见的有 BPE(Byte-Pair Encoding)，WordPiece(Character-Level BPE)，Byte-level BPE 等

tokenizer 库提供了常见的训练与处理方法，大体上分为以下几个组件：

1. Models（模型）
   作用：这一组件通常包含用于文本编码的模型，如 WordPiece 模型或 Byte Pair Encoding（BPE）。这些模型负责将词汇或字符序列转换为模型可以理解的内部表示。
   功能：它定义了如何将输入文本分解为更小的单元（如子词或字节），并将这些单元映射到词汇表中的索引。
2. Normalizers（规范化器）
   作用：在文本被 Tokenize 之前，Normalizer 负责对文本进行预处理，以确保文本的一致性和标准化。
   功能：常见的规范化操作包括转换为小写、去除或替换特殊字符、标准化空白符、进行词干提取等。
3. Pretokenizer（预分词器）
   作用：Pretokenizer 是在 Normalizer 之后、Tokenization 之前的一个步骤，用于将文本切分成更小的片段。
   功能：它可以执行如按空白符分词、处理 URL、电子邮件地址、特定符号等任务，为 Tokenization 过程准备更细粒度的输入。
4. PostProcessor（后处理器）
   作用：PostProcessor 用于在 Tokenization 完成后对生成的 Token 序列进行进一步的处理。
   功能：这可能包括添加特殊标记（如开始和结束标记）、进行子词重组（如将多个 WordPiece 标记合并为一个单词）、处理序列长度限制等。
5. Decoder（解码器）
   作用：Decoder 负责将模型的输出（通常是 Token ID 序列）转换回可读的文本格式。
   功能：它将 Token ID 映射回原始词汇表中的单词或字符，可能还需要处理特殊标记和子词单元的合并。

训练完整流程如下，其中`tokenizer_train.jsonl`来自[网址](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

```
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import json
import os
from transformers import AutoTokenizer

def train_tokenizer():
    # 训练数据读取
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    data_iter = read_texts_from_jsonl("./dataset/tokenizer_train.jsonl")

    # 初始化tokenizer，配置各组件(Nomalizer、Pretokenizer、PostProcessor、Decoder)
    tokenizer = Tokenizer(model=models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()


    # 配置训练参数
    special_tokens = ["<unk>","<s>","</s>"]

    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 训练
    tokenizer.train_from_iterator(iterator=data_iter,trainer=trainer)

    '''
    tokenizer_config.json表明如何从文本到token和token到文本
    tokenizer.json表面如何从输入到文本
    '''

    # 保存tokenizer(生成tokenizer.json)
    tokenizer_dir = "./tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    # 手动创建配置文件(tokenizer_config.json)
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")

def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/code/mymodel/model/tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))


    input_ids = model_inputs['input_ids']
    response = tokenizer.decode(input_ids)
    print('decoder和原始文本是否一致：', response == new_prompt)

if __name__=="__main__":
    train_tokenizer()
    eval_tokenizer()

```
