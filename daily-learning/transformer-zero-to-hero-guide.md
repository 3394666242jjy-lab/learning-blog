# Transformer 零基础深入学习指南：从神经网络到 2026 前沿

> **阅读提示**：本文是我在大二下学期系统学习 Transformer 家族时的笔记整理。从零基础的神经网络概念出发，一路梳理到 2026 年的前沿架构（MoE、长上下文外推、推理模型）。如果你也是刚入门深度学习，希望这份"地图"能帮你少走一些弯路。
> 
> **关键词**：Transformer | Attention | 大语言模型 | MoE | 推理模型 | 学习路线

---

## 目录

1. [为什么写这篇博客？](#1-为什么写这篇博客)
2. [第一章：从感知机到 RNN——序列建模的困境](#2-第一章从感知机到-rnn序列建模的困境)
3. [第二章：Attention 机制——带权求和的艺术](#3-第二章attention-机制带权求和的艺术)
4. [第三章：Transformer 架构完整解析](#4-第三章transformer-架构完整解析)
5. [第四章：2017→2026 七大进化主线](#5-第四章20172026-七大进化主线)
6. [第五章：五大模型家族与范式跃迁](#6-第五章五大模型家族与范式跃迁)
7. [第六章：前沿技术专题（MoE / RLHF / CoT）](#7-第六章前沿技术专题moe--rlhf--cot)
8. [第七章：我的学习路线与资源推荐](#8-第七章我的学习路线与资源推荐)
9. [批判与反思：这篇文档的局限](#9-批判与反思这篇文档的局限)
10. [参考文献](#10-参考文献)

---

## 1. 为什么写这篇博客？

进入医疗大模型实验室后，导师扔给我一堆论文：Attention Is All You Need、BERT、GPT、LLaMA、DeepSeek-R1……我一开始是懵的——这些模型名字听起来很酷，但它们之间到底是什么关系？为什么今天用 RoPE，明天又用 GQA？

我意识到，**孤立地读每一篇论文就像孤立地背每一个单词，没有语法和语境，永远造不出句子**。所以我花了几天时间，把 Transformer 从诞生到 2026 年的演进梳理成一张"族谱"，这篇博客就是这张族谱的笔记版。

**目标读者**：
- 刚接触深度学习，听说过 Transformer 但不知道它内部怎么运转的同学
- 想申请实验室/读研，面试时可能被问到 "讲讲 Attention 机制" 的同学
- 已经会用 HuggingFace 调包，但想理解 "为什么这样设计" 的同学

---

## 2. 第一章：从感知机到 RNN——序列建模的困境

### 2.1 神经网络的本质：加权求和 + 非线性激活

神经网络（Neural Network）模仿的是人脑神经元。最简单的**感知机（Perceptron）**诞生于 1958 年，只有输入层和输出层，只能解决线性可分问题（比如用一条直线区分苹果和橙子）。

当问题变复杂，我们需要**隐藏层**和**激活函数**（ReLU、Sigmoid、Tanh）。激活函数的作用是引入非线性——没有它，无论堆多少层，网络本质上还是一个线性模型。

**学习三件套**：
1. **前向传播**：数据从输入走到输出，得到预测结果
2. **计算损失**：对比预测和真实答案，算出误差
3. **反向传播**：误差从输出传回输入，调整每个连接的权重

### 2.2 RNN 与 LSTM：给网络加"记忆"

标准神经网络（前馈网络）有个致命缺陷：**它把输入当成一个袋子，不考虑顺序**。你先输入"我"再输入"喜欢"，和先输入"喜欢"再输入"我"，网络的处理方式完全一样。

**循环神经网络（RNN, Recurrent Neural Network）**解决了这个问题。它引入了一个**隐藏状态（Hidden State）**，就像人的短期记忆：每读到一个新词，就结合之前读过的内容来理解。

但 RNN 有个**长期依赖问题**。想象你读一本 500 页的书，读到最后一页时，你对第一页内容的记忆还剩下多少？信息在传递过程中会逐层"稀释"。

1997 年提出的**长短期记忆网络（LSTM, Long Short-Term Memory）**用三个"门"来控制信息流动：
- **遗忘门**：决定忘记多少旧信息
- **输入门**：决定存入多少新信息
- **输出门**：决定输出多少信息

LSTM 还引入了一条**细胞状态（Cell State）**传送带，让信息可以相对 unchanged 地流动。LSTM 在很长一段时间内都是序列建模的标准答案。

### 2.3 RNN 的两大瓶颈

然而，LSTM 并没有从根本上解决问题。

**瓶颈一：梯度消失与梯度爆炸**

反向传播时，梯度要乘以激活函数的导数。如果导数小于 1，梯度会越乘越小，最终几乎为 0——**梯度消失（Vanishing Gradient）**。反之，如果导数大于 1，梯度会指数级爆炸——**梯度爆炸（Exploding Gradient）**。序列越长，这个问题越严重。实践表明，当序列长度超过 100 个词时，LSTM 性能明显下降。

**瓶颈二：串行计算**

RNN 每个时刻的计算都依赖前一个时刻的隐藏状态，所以必须**按顺序算**。处理 1000 个词的文本需要 1000 步串行计算，GPU 的并行计算优势被完全浪费。

> **思考**：有没有一种方法，既能捕捉长距离依赖，又能并行计算？

### 2.4 Attention 的萌芽（2014）

在 Transformer 出现之前，研究者已经意识到 RNN 的局限性。2014 年，Dzmitry Bahdanau 等人在机器翻译中引入了 **Attention 机制**：生成目标语言的每个词时，模型不应该只依赖 RNN 的最后一个隐藏状态，而应该**"回头看"源语言的所有词**，根据相关性给予不同的关注度。

Attention 极大提升了翻译质量，但此时它只是 RNN 的"附件"——RNN 仍然负责主要的序列处理。真正的突破在 2017 年到来。

---

## 3. 第二章：Attention 机制——带权求和的艺术

### 3.1 为什么需要 Attention？

考虑这两个句子：
- "苹果很甜。" → 苹果是水果
- "苹果发布了新手机。" → 苹果是公司

人类能区分这两个"苹果"，是因为我们会结合上下文判断。Attention 机制就是让模型具备这种能力：**在处理每个词时，动态地关注序列中最相关的其他词**。

Attention 的本质可以概括为四个字：**带权求和**。

### 3.2 核心过程：三步走

**第一步：计算相似度（打分）**

当模型要理解"苹果"时，它会计算"苹果"与句子中每个其他词的相关程度。"苹果"与"发布"的关联度可能很高，与"了"的关联度可能很低。

**第二步：归一化为权重（Softmax）**

把相似度分数通过 Softmax 转换成概率分布（所有权重之和为 1）。每个词都得到一个 0~1 之间的权重值。

**第三步：加权求和**

用这些权重对所有词的向量表示进行加权求和，得到"苹果"的新表示。这个新表示融合了上下文信息，使得同一个词在不同句子中含义不同。

### 3.3 Q、K、V：信息检索的三剑客

Attention 中有三个核心概念，理解它们就理解了 Attention 的骨架：

| 角色 | 含义 | 类比 |
|------|------|------|
| **Q (Query, 查询)** | "我想找什么信息？" | 你在图书馆问管理员："我想找漫威相关的书" |
| **K (Key, 键)** | "我是什么标签？" | 每本书的标签（科幻、漫画、动作） |
| **V (Value, 值)** | "我真正的内容是什么？" | 书里面的实际内容 |

对于输入序列中的每个词，模型通过三个不同的**可学习投影矩阵** $W_Q, W_K, W_V$ 生成对应的 Q、K、V：

$$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$

### 3.4 Self-Attention：自己关注自己

在 Transformer 中主要使用**自注意力（Self-Attention）**。意思是：Q、K、V 都来自**同一个序列**。序列中的每个词都会去"询问"序列中的所有词（包括自己），计算彼此之间的关联度。

以"苹果发布了新手机"为例：
1. 将每个词转换为向量（词嵌入）
2. 对每个词向量，通过三个线性变换得到 Q、K、V
3. "苹果"的 Q 与所有词的 K 做匹配：与自己的 K 匹配度高（自关注），与"发布"的 K 匹配度高（动作关系），与"了"的 K 匹配度低（虚词不重要）
4. 匹配分数通过 Softmax 变成权重
5. 用权重对所有词的 V 加权求和，得到"苹果"的新表示

**Self-Attention 的三大优势**：
1. **捕捉长距离依赖**：无论两个词相距多远，都可以直接计算关系，不需要一步步传递
2. **可并行计算**：所有词的注意力计算相互独立，GPU 可以并行处理
3. **可解释性强**：注意力权重可以可视化，看到模型关注了哪些词

### 3.5 Multi-Head Attention：多角度看问题

单一的注意力机制可能不足以捕捉词语之间多样化的关系。Transformer 提出了**多头注意力（Multi-Head Attention, MHA）**：与其只计算一次注意力，不如并行计算多组注意力，每组关注不同的方面。

想象一个团队分析"小明去了学校"：
- 分析师 A 关注语法结构（"小明"是主语，"去了"是谓语）
- 分析师 B 关注语义关系（"学校"是"去了"的目的地）
- 分析师 C 关注情感色彩（中性陈述）

每个分析师独立工作，最后综合大家的观点。技术上，MHA 将输入向量维度分成 $h$ 组（通常 $h=8$ 或 $16$），每组独立计算 Self-Attention，最后拼接并通过线性变换。

### 3.6 缩放点积注意力：数学公式拆解

Attention 的核心公式：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(Q \cdot W_i^Q, K \cdot W_i^K, V \cdot W_i^V)$$

**拆解**：
1. **$QK^T$**：计算相似度。结果是一个 $n 	imes n$ 的矩阵，每个元素 $(i,j)$ 表示第 $i$ 个词和第 $j$ 个词的相似度
2. **$\div \sqrt{d_k}$**：缩放。$d_k$ 是每个注意力头的维度。除以 $\sqrt{d_k}$ 防止点积结果过大，导致 Softmax 输出极端（接近 0 或 1），梯度变小，训练变慢
3. **Softmax**：归一化，转换为概率分布
4. **$	imes V$**：加权求和，得到最终输出

---

## 4. 第三章：Transformer 架构完整解析

### 4.1 诞生：《Attention Is All You Need》（2017）

2017 年 6 月，Google 发表了一篇标题极具宣言性的论文——**《Attention Is All You Need》**。它提出了一个大胆的观点：**我们可以完全抛弃 RNN，只用 Attention 机制来构建模型**。

Transformer 的三大革命性贡献：
1. **完全并行化**：序列中所有位置同时计算注意力，训练效率大幅提升
2. **长距离依赖不再是问题**：任意两个位置之间的"距离"都是 1，直接通过 Attention 建立联系
3. **模块化设计**：Encoder-Decoder 结构清晰，易于扩展，为 BERT、GPT 等后续模型奠定基础

### 4.2 Encoder-Decoder 结构总览

原始 Transformer 采用**编码器-解码器（Encoder-Decoder）**结构，最初用于机器翻译。

**Encoder（编码器）**：由 $N$ 个相同层堆叠（原始论文 $N=6$）。每层包含两个子层：
1. **Multi-Head Self-Attention**：让输入序列中的每个位置关注所有其他位置，捕捉全局依赖
2. **Feed-Forward Network (FFN)**：对每个位置独立进行非线性变换，增加表达能力

每个子层之后都有**残差连接（Residual Connection）**和**层归一化（Layer Normalization）**。

**Decoder（解码器）**：同样由 $N$ 层堆叠。每层包含三个子层：
1. **Masked Multi-Head Self-Attention**：与 Encoder 类似，但增加**掩码（Mask）**，防止当前位置关注到未来的位置（生成时不能"偷看"后面的内容）
2. **Multi-Head Cross-Attention**：让 Decoder 关注 Encoder 的输出，相当于"翻译时回头看原文"
3. **Feed-Forward Network**

**类比：同声传译员的工作流程**
- **Encoder** = 仔细阅读英文原文，标记关键词和句子结构
- **Decoder** = 开始说中文，每说一个新词时：
  1. 回顾自己已经说了什么（Masked Self-Attention）
  2. 回头看原文确保没有遗漏（Cross-Attention）
  3. 在大脑中组织语言（FFN）

### 4.3 位置编码（Positional Encoding）

Attention 有一个特点：**它对输入的顺序不敏感**。无论"我喜欢你"还是"你喜欢我"，只要词相同，Attention 计算出的结果都一样（假设 Q、K、V 相同）。

但语言中顺序至关重要。"我打你"和"你打我"意思完全不同。为了让模型知道词语的位置，Transformer 引入了**位置编码（Positional Encoding）**。

原始 Transformer 使用**正弦/余弦函数**生成位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

选择正弦/余弦的原因：
1. **唯一性**：每个位置都有唯一编码
2. **有界性**：值域在 [-1, 1] 之间
3. **相对位置关系**：对于固定偏移量 $k$，$PE_{pos+k}$ 可以通过 $PE_{pos}$ 的线性变换得到
4. **外推性**：可以扩展到训练时未见过的更长序列

位置编码与词嵌入相加，作为模型的真正输入。后续研究提出了可学习的位置嵌入（BERT）、旋转位置编码 RoPE（LLaMA）等改进方案。

### 4.4 残差连接与层归一化

Transformer 是深层网络（原始 12 层，现代版本可能 96 层甚至更多）。训练深层网络面临梯度消失和网络退化。

**残差连接（Residual Connection）**：在子层的输入和输出之间加一条"捷径"，让输出等于输入加上子层的变换结果：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$
这保证了即使网络很深，梯度也能顺利从深层传回浅层。

**层归一化（Layer Normalization, LN）**：对每个样本的特征维度进行归一化，使其均值为 0、方差为 1，稳定训练过程。

原始 Transformer 使用 **Post-LN**（子层之后做归一化），但后续研究发现 **Pre-LN**（子层之前做归一化）让训练更稳定，尤其是模型很深时。现代模型（GPT-3、LLaMA）普遍采用 Pre-LN 或其变体 **RMSNorm**（去掉减均值操作，只保留缩放，计算更快）。

### 4.5 前馈神经网络（FFN）

每个 Encoder 和 Decoder 层中，除了 Attention 子层外，还有一个 **FFN（Feed-Forward Network）** 子层。FFN 对 Attention 的输出进行进一步的非线性变换，增强表达能力。

原始 Transformer 的 FFN 由两个线性变换层和一个 ReLU 激活函数组成：

$$\text{FFN}(x) = \max(0, x \cdot W_1 + b_1) \cdot W_2 + b_2$$
FFN 的特点是**对每个位置独立处理，位置之间没有信息交互**。信息交流的工作完全交给 Attention，这种分工明确是 Transformer 成功的关键之一。

FFN 的中间层维度通常是输入/输出维度的 4 倍（例如模型维度 512，FFN 中间维度 2048）。

### 4.6 Masked Self-Attention 与 Causal Masking

Decoder 中的 Self-Attention 有一个特殊变体——**Masked Self-Attention（掩码自注意力）**。

在生成任务中，Decoder 是一个词一个词地生成输出的。当生成第 $i$ 个词时，模型只能看到已经生成的词（位置 1 到 $i-1$），不能"偷看"尚未生成的词（位置 $i+1$ 及之后）。

Mask 的作用是把"未来位置"的注意力分数设为极大的负数（$-\infty$），使得 Softmax 之后这些位置的权重为 0。

这种机制称为 **Causal Masking（因果掩码）**，保证了生成过程的有序性和因果性。

---

## 5. 第四章：2017→2026 七大进化主线

从 2017 年 Transformer 诞生至今的 9 年间，架构经历了大量改进。理解这 7 条进化线，就像拿到了一张"技术族谱"，看到任何新论文都能快速定位它的位置。

### 5.1 归一化位置的演进：Post-LN → Pre-LN → RMSNorm

| 策略 | 位置 | 优点 | 缺点 | 代表模型 |
|------|------|------|------|----------|
| **Post-LN** | 残差后 | 前向传播稳定 | 深层梯度消失，训练不稳 | 原版 Transformer |
| **Pre-LN** | 子层前 | 训练稳定，易于扩展 | 注意力可能过自信 | GPT-2/3, LLaMA |
| **Pre-LN + QK-Norm** | 子层前 + QK 归一化 | 训练稳定 + 注意力均衡 | 计算略增 | Gemma 3/4, Qwen3 |
| **RMSNorm** | 子层前 | 计算更快 | 无中心化 | LLaMA, DeepSeek |

- **Post-LN**：原版配置，小模型工作良好，大模型训练不稳定
- **Pre-LN**：先对输入做 LN，再通过子层，最后与输入相加。极大提升训练稳定性，使得 GPT-3 能堆到 175B 参数
- **RMSNorm**：LayerNorm 的简化版，去掉减均值，只保留缩放。LLaMA、Qwen 等广泛采用

### 5.2 注意力共享的演进：MHA → MQA → GQA → MLA

这条线的核心目标是**降低推理时的 KV-Cache 内存占用**。

| 机制 | K/V 共享方式 | KV-Cache | 表达能力 | 代表模型 |
|------|-------------|----------|----------|----------|
| **MHA** | 每头独立 | 最大 | 最强 | 原版 Transformer, BERT |
| **MQA** | 全部共享 | 最小 | 较弱 | 早期实验模型 |
| **GQA** | 分组共享 | 中等 | 良好 | LLaMA-3, Mistral |
| **MLA** | 潜在空间压缩 | 极小 | 强 | DeepSeek-V3/V4 |

- **MHA（Multi-Head Attention）**：每个头都有独立的 K 和 V。表达能力最强，但 KV-Cache 占用最大。100K token 的长上下文可能占用数 GB 显存。
- **MQA（Multi-Query Attention）**：所有头共享同一组 K 和 V。大幅降低缓存，但损害表达能力。
- **GQA（Grouped-Query Attention）**：折中方案。将多个 Query 头分成若干组，每组共享一组 K/V。例如 32 个头分为 8 组，KV-Cache 降为 MHA 的 1/4，性能几乎无损。2024 年 LLaMA-3 采用后，几乎所有新旗舰都跟进。
- **MLA（Multi-head Latent Attention）**：DeepSeek 提出，将 K 和 V 压缩到一个低维"潜在空间"，进一步减少 KV-Cache。DeepSeek-V4 Pro 还引入 CSA + HCA 混合注意力，在 1M token 上下文中将单 token 计算量降到早期的 27%。

### 5.3 位置编码的演进：正余弦 → RoPE → 长上下文外推

- **正弦/余弦绝对位置编码（2017）**：原版 Transformer 使用。不擅长外推到训练长度之外的序列——如果只在 4K 长度上训练，处理 100K 文本时性能急剧下降。
- **RoPE（Rotary Position Embedding, 旋转位置编码，2021）**：将位置信息编码到 Q/K 向量的**旋转角度**中。天然支持任意长度外推，是现代长上下文模型的基础（LLaMA、Qwen）。
- **iRoPE / 滑动窗口（2026）**：LLaMA 4 使用 iRoPE（interleaved RoPE），通过交织方式更好处理超长上下文。MiniMax Text-01 使用 Lightning Attention 和 Softmax 注意力的混合方案，支持 **4M token** 的推理上下文。

### 5.4 激活函数与 FFN 的演进：ReLU → GELU → SwiGLU

- **ReLU（2017）**：原版使用。$	ext{ReLU}(x) = \max(0, x)$。简单高效，但负数区域完全归零，可能导致"神经元死亡"。
- **GELU（2018）**：高斯误差线性单元，曲线更平滑，负数区域仍有微小梯度。BERT 采用后性能提升。
- **SwiGLU（2022）**：结合 Swish 激活函数和门控线性单元（GLU）。PaLM 论文证明 SwiGLU 在相同参数量下表现优于 GELU 和 ReLU。此后 LLaMA、Qwen、DeepSeek 等模型都采用 SwiGLU。

### 5.5 架构形态的演进：Encoder-Decoder → Decoder-Only

| 架构 | 结构 | 预训练任务 | 优势领域 | 代表模型 |
|------|------|-----------|----------|----------|
| **Encoder-Only** | 仅编码器 | 掩码语言模型（MLM） | 文本理解、分类 | BERT, RoBERTa |
| **Decoder-Only** | 仅解码器 | 自回归语言模型 | 文本生成、通用任务 | GPT, LLaMA, Qwen |
| **Encoder-Decoder** | 编码器+解码器 | Text-to-Text | 翻译、摘要 | T5, BART |

- **Encoder-Decoder（2017）**：适用于序列到序列任务（如翻译）。编码器处理输入，解码器生成输出。
- **Encoder-Only（2018, BERT）**：只保留编码器，适用于理解类任务（分类、填空、问答）。通过掩码语言模型（MLM）预训练。
- **Decoder-Only（2018, GPT 系列）**：只保留解码器，适用于生成类任务。使用自回归方式（逐词生成）预训练。后续研究表明 Decoder-Only 不仅擅长生成，通过适当训练也能在理解任务上表现出色。

**当今几乎所有大语言模型（GPT、LLaMA、Qwen、DeepSeek）都采用 Decoder-Only 架构。**

### 5.6 稀疏化革命：混合专家模型（MoE）的复兴

**MoE（Mixture of Experts）**的核心思想是：**模型很大，但每次只激活一小部分**。

传统"稠密模型"（Dense Model）处理每个输入时都会使用全部参数。而 MoE 将 FFN 层替换为多个"专家"网络，并引入一个"门控网络"（Gating Network）决定每个输入应该激活哪些专家。

**类比**：想象一家大型医院，里面有各种专科医生（专家）。当一个病人来看病时，分诊护士（门控网络）根据症状把病人引导到最合适的专科。心脏病人去看心脏科，骨折病人去看骨科——不需要每个病人都看所有医生。

2024 年被称为"MoE 复兴年"：DeepSeek-V3（671B 总参数 / 37B 激活）、Mixtral、Qwen-MoE、Gemini 1.5 几乎同时切换到 MoE 路线。2026 年，GLM-5 将激活比例压到 **5.4%**（744B 总参数 / 40B 激活），每 token 只激活几十亿参数，但效果接近稠密旗舰。

**MoE 的优势**：用更少的计算获得更大的模型容量。一个"万亿参数"的 MoE 模型，实际推理计算量可能和"百亿参数"的稠密模型相当，但能力远超后者。

**代价**：需要更多显存存储所有专家参数，以及更复杂的训练基础设施（路由、负载均衡、通信）。

### 5.7 推理能力的进化：从 RLHF 到思维链（CoT）

- **RLHF（Reinforcement Learning from Human Feedback, 2022）**：基于人类反馈的强化学习，是 GPT-3.5 → ChatGPT 的关键一步。让模型学会人类偏好——什么回答是好的、安全的、符合伦理的。
- **RL on CoT（2024, OpenAI o1）**：将强化学习应用到**思维链（Chain of Thought, CoT）**上。CoT 不再是简单的 prompt 技巧，而是训练时的优化信号。模型被鼓励"多想一会儿"，在给出最终答案前先进行详细推理。
- **Adaptive Thinking（2025-2026）**：各家模型把"思考"做成了可调参数：
  - OpenAI：`reasoning_effort`（low/medium/high/xhigh/heavy）
  - Anthropic：`adaptive thinking`
  - Google：`thinking_level`
  - DeepSeek-V4：`Think Max`

方向一致：**简单题快答，难题慢想**。这种"自适应思考"正在模糊训练时和推理时的边界。

---

## 6. 第五章：五大模型家族与范式跃迁

### 6.1 Open-US/EU 开源系

学术界和工业界协作的主战场，完全开源（代码、数据、checkpoint 全部公开）。

- **Pythia（2023, EleutherAI）**：规模从 70M 到 12B，完全可复现——训练数据、代码、所有中间 checkpoint 全部公开
- **LLaMA（Meta, 2023-2024）**：LLaMA-1 引发开源大模型浪潮，LLaMA-2 增加商用授权，LLaMA-3 采用 GQA。是许多开源模型的基础（Alpaca、Vicuna 等基于 LLaMA 微调）
- **Mistral（2023, 法国）**：Mistral 7B 以小巧体型实现接近 LLaMA-2 13B 的性能。Mixtral 8x7B 是早期采用 MoE 的开源模型之一
- **OLMo（2024-2025, Allen AI）**：主打"完全透明"——不仅权重开源，训练数据、代码、日志、所有 checkpoint 全部公开
- **Gemma（Google, 2024-2025）**：基于 Gemini 技术，Gemma 3/4 采用 RMSNorm + QK-Norm

### 6.2 Open-CN 开源中国队

2024-2026 年，中国开源模型在多个方向实现范式创新的先发。

- **Qwen（阿里巴巴）**：全球最受欢迎的开源模型之一。Qwen2.5/3 不断优化，Qwen3.6-27B 使用 Gated DeltaNet 在 27B 稠密参数上接近 397B MoE 的表现
- **DeepSeek（深度求索）**：DeepSeek-V3 的 MLA 注意力机制大幅降低 KV-Cache；V4 Pro 引入 CSA + HCA 混合注意力；DeepSeek-R1 的开源让推理模型训练方法公开化，2025 年 9 月登上 Nature 封面
- **Kimi（月之暗面）**：创始人杨植麟是 XLNet 和 Transformer-XL 的共同作者。Kimi K2.6 使用 MLA + 1T MoE，将 256K 上下文推到与 GPT-5.4 同档
- **GLM（智谱）**：GLM-5 在华为昇腾芯片和 MindSpore 框架上完成全栈训练，是首批不依赖 NVIDIA 的 frontier 模型之一。激活比例压到 5.4%，是 MoE 效率的标杆

### 6.3 闭源旗舰系

架构细节不公开，但从技术报告和反推证据看，同样遵循 7 条进化线，只是在参数规模、数据规模和训练算力上更有优势。

- **GPT 系列（OpenAI）**：GPT-1/2/3 证明堆参数路线的可行性，GPT-4 引入多模态，o1/o3 系列引领推理模型
- **Claude 系列（Anthropic）**：以安全性和长上下文著称。Claude Opus 4.7 引入 adaptive thinking
- **Gemini 系列（Google）**：原生多模态设计（文本+图像+音频+视频）。Gemini 2.5 Thinking 跟进推理模型路线

### 6.4 五个范式跃迁时刻

1. **PaLM 540B（2022, Google）**：SwiGLU + RoPE + Parallel Layers 的组合成为现代 Transformer 的"工业范本"
2. **GQA 普及（2024, LLaMA-3）**：同质量下推理 KV-Cache 降 4-8 倍，一出现几乎所有新旗舰都跟进
3. **Sparse MoE 复兴（2024）**：一年内 DeepSeek-V3、Mixtral、Qwen-MoE、Gemini 1.5 集体切换到 MoE。OpenAI 2025 年 8 月用 GPT-OSS（117B/5.1B 激活）重新加入开源队列——自 GPT-2（2019）后首次发布开源权重
4. **o1 RL on CoT（2024, OpenAI）**：CoT 从 prompt 技巧变成 train-time 优化信号。DeepSeek-R1、Claude 3.7 Sonnet、Gemini 2.5 Thinking 一年内全部跟进
5. **开源中国队多范式先发（2024-2026）**："闭源美国旗舰单方向领跑"的叙事已经不准。开源生态在多个细分方向反向定义议程：DeepSeek 的 MLA/CSA+HCA、GLM-5 的国产芯片训练、Qwen 的 Gated DeltaNet、Kimi 的长上下文 MLA

---

## 7. 第六章：前沿技术专题（MoE / RLHF / CoT）

### 7.1 混合专家模型（MoE）深度解析

#### 核心组件

1. **专家网络（Experts）**：每个专家通常是一个标准的 FFN。一组专家（如 8 个或 16 个）并行排列。注意："专家"不是在特定领域（如"数学专家"）上的专业化——更准确地说，每个专家学习了处理特定上下文中的特定 token 模式。
2. **门控网络（Gating Network / Router）**：小型神经网络，决定每个输入 token 应该被发送到哪些专家。接收输入 token 的向量表示，输出概率分布（通过 Softmax）。
3. **负载均衡机制**：简单的门控网络容易"偷懒"——总是把任务分配给少数几个学得快的专家。MoE 引入**负载均衡损失（Load Balancing Loss）**，鼓励 token 均匀分配给所有专家。

#### 工作流程（以"写一段关于猫的加速度的科普文字"为例）

1. 文本被转换为向量，传入 MoE 层
2. 门控网络分析输入向量，给每个专家打分："科普文本生成专家"0.6 分、"物理公式专家"0.3 分、"文学修辞专家"0.1 分
3. 选择分数最高的 Top-K 个专家（通常 K=2）。只有被选中的专家参与计算
4. 被选中的专家独立处理输入，各自产生输出
5. 根据门控分数，对各专家的输出进行加权求和，得到最终输出

#### 优势与挑战

| 优势 | 挑战 |
|------|------|
| 预训练速度更快（相同计算量下可训练更大模型） | 需要大量显存存储所有专家参数 |
| 推理速度更快（只激活部分专家） | 微调时容易过拟合 |
| 模型容量大，性能强 | 负载均衡需要精心设计 |
| 天然适合多任务 | 通信开销大（分布式部署时） |

### 7.2 人类反馈强化学习（RLHF）

RLHF 是让大模型从"会回答"变成"回答得让人满意"的关键技术。

**为什么需要 RLHF？**

传统监督微调（SFT）只能让模型学会"输入-输出"的映射——给定问题，输出"标准答案"。但 SFT 无法教会模型**偏好**：对于同一个问题，哪种回答方式更好？

例如问"如何学习大模型"：
- 回答 A："多学习，多实践。"（正确但空洞）
- 回答 B："如果你有编程基础，可以从 PyTorch 开始，然后阅读 Transformer 论文……"（具体、可操作）

SFT 无法区分 A 和 B 的优劣，因为它只是让模型模仿训练数据。RLHF 通过引入人类偏好来解决这个问题。

**RLHF 三步骤**：

1. **监督微调（SFT）**：先用高质量人工标注数据对预训练模型进行微调，让模型学会基本的对话格式和回答风格
2. **训练奖励模型（Reward Model）**：让模型对同一个问题生成多个回答，由人类标注员排序（哪个更好）。用这些排序数据训练一个"奖励模型"——预测人类会喜欢哪个回答
3. **强化学习优化（PPO）**：使用 PPO（Proximal Policy Optimization）算法，让语言模型生成回答，奖励模型打分，语言模型根据分数调整策略，朝着高分方向优化

**类比**：SFT 相当于老师给标准答案，学生照抄；RLHF 相当于老师让学生自己答题，然后给反馈——"这个回答太笼统了，扣 2 分""这个例子举得好，加 3 分"。

**DPO（Direct Preference Optimization）**：2023 年斯坦福大学提出，直接用人类偏好数据优化语言模型，无需单独的奖励模型和复杂的强化学习算法。更简单高效，已成为许多模型的首选对齐方法。

### 7.3 推理模型与思维链（Chain of Thought）

2024 年以来，大模型领域的重要趋势是从"快思考"（快速回答问题）向"慢思考"（先推理再回答）的转变。

**什么是 CoT？**

思维链（Chain of Thought, CoT）是指模型在给出最终答案之前，先生成一段详细的推理过程。

**例子**：
- 问题：一个农场有鸡和兔共 35 只，脚共 94 只。鸡和兔各有多少只？
- 标准回答：鸡有 23 只，兔有 12 只。
- CoT 回答：假设全是鸡，那么应该有 35×2=70 只脚。实际有 94 只脚，多出了 94-70=24 只脚。每把一只鸡换成兔，脚增加 2 只，所以需要换 24÷2=12 次。因此兔有 12 只，鸡有 35-12=23 只。

CoT 最早是作为一种 prompt 技巧（让模型 "Let's think step by step"）被发现的。但 OpenAI 的 o1 模型把 CoT 提升到了新高度——在训练时用强化学习优化 CoT 的质量，让模型学会"先仔细思考再回答"。

**推理模型的训练方法（DeepSeek-R1 揭示）**：

1. **冷启动**：先收集少量高质量 CoT 数据，对基础模型进行 SFT
2. **强化学习**：让模型生成大量带有推理过程的回答，对正确答案给予奖励。模型通过试错学会：更长的思考、自我检查、尝试不同方法，可以提高正确率
3. **自我进化**：模型会自动发展出各种推理策略：反思（检查自己的推理是否正确）、回溯（发现错误后返回重新推理）、分解（把复杂问题拆成简单子问题）等

**Adaptive Thinking（自适应思考）**：

2025-2026 年，各家模型推出可调节的"思考深度"：

| 厂商 | 机制名称 | 档位 |
|------|----------|------|
| OpenAI | `reasoning_effort` | low/medium/high/xhigh/heavy |
| Anthropic | `adaptive thinking` | 模型自决思考深度 |
| Google | `thinking_level` + `thinking_budget` | L/M/H 三档 |
| DeepSeek | `Think Max` | 三档调节 |
| Qwen | `thinking mode` / `Thinking Preservation` | 多档可调 |

核心思想：**简单的问题快速回答，复杂的问题深入思考**。这不仅提升用户体验，也优化了计算资源的利用。

### 7.4 2026 年主流 Transformer 架构配方总结

经过 9 年进化，当前主流模型的"标准配方"已经趋于稳定：

| 组件 | 主流选择 | 说明 |
|------|----------|------|
| 基础架构 | **Decoder-Only** | Encoder-Only 和 Encoder-Decoder 逐渐边缘化 |
| 归一化 | **Pre-LN / RMSNorm + QK-Norm** | 训练稳定，防止注意力过自信 |
| 注意力 | **GQA / MLA** | 大幅降低 KV-Cache |
| 位置编码 | **RoPE / iRoPE** | 支持长上下文外推 |
| 激活函数 | **SwiGLU** | PaLM 验证的最佳选择 |
| 架构类型 | **Dense / MoE** | MoE 逐渐成为主流选择 |
| 后训练 | **SFT + RLHF / DPO** | 对齐人类偏好 |
| 推理 | **CoT / Adaptive Thinking** | 思考深度可调 |

---

## 8. 第七章：我的学习路线与资源推荐

### 8.1 零基础学习路线（四阶段）

如果你是完全的初学者，建议按以下路线逐步深入：

**第一阶段：打好基础（1-2 个月）**
1. Python 编程：熟练掌握基础语法，这是所有深度学习的基础
2. 线性代数与概率论：理解向量、矩阵、概率分布等基本概念。不需要很深，但要知道它们是什么
3. 机器学习基础：了解监督学习、无监督学习、梯度下降等基本概念。推荐 Andrew Ng 的《Machine Learning Specialization》

**第二阶段：深度学习入门（1-2 个月）**
1. 神经网络基础：理解前馈神经网络、反向传播、激活函数。推荐 3Blue1Brown 的神经网络系列视频
2. 深度学习框架：学习 PyTorch，能够搭建简单的神经网络
3. CNN 与 RNN：了解卷积神经网络（用于图像）和循环神经网络（用于序列），理解它们的基本原理和局限性

**第三阶段：Transformer 深入（2-3 个月）**
1. Attention 机制：彻底理解 Self-Attention 和 Multi-Head Attention 的原理，能够手写代码实现
2. Transformer 架构：阅读《Attention Is All You Need》原文，理解 Encoder-Decoder 的每个组件
3. 预训练模型实践：使用 Hugging Face Transformers 库，加载 BERT/GPT 等模型进行微调实验
4. 阅读原始论文：阅读 BERT、GPT-2/3、LLaMA、DeepSeek 等关键论文，理解技术演进

**第四阶段：前沿探索（持续）**
1. MoE 技术：阅读 DeepSeek-MoE、Mixtral 等论文，理解稀疏激活的原理
2. 长上下文技术：学习 RoPE、YaRN、NTK-aware 扩展等长上下文外推方法
3. 推理模型：阅读 DeepSeek-R1 论文，了解 RL on CoT 的训练方法
4. 动手实践：尝试微调开源模型，参与开源项目，或复现经典论文

### 8.2 核心要点回顾（一张图）

**从神经网络到 Transformer 的进化线**：

感知机（1958）→ 多层神经网络（1986+）→ RNN（1986）→ LSTM（1997）→ Attention（2014）→ Transformer（2017）

每次演进都是为了解决前一代的问题：RNN 解决序列建模，LSTM 缓解长期依赖，Attention 让模型选择性关注，Transformer 彻底抛弃串行计算。

**Transformer 的 7 条进化主线（2017-2026）**：
1. 归一化：Post-LN → Pre-LN → RMSNorm + QK-Norm
2. 注意力共享：MHA → MQA → GQA → MLA
3. 位置编码：正余弦 → RoPE → iRoPE / 滑动窗口
4. 激活函数：ReLU → GELU → SwiGLU
5. 架构形态：Encoder-Decoder → Decoder-Only
6. 稀疏化：Dense → MoE（混合专家模型）
7. 推理能力：RLHF → RL on CoT → Adaptive Thinking

**五大模型家族**：
- **Open-US/EU 开源系**：LLaMA、Mistral、OLMo、Gemma——学术友好的完全开源模型
- **Open-CN 开源中国队**：Qwen、DeepSeek、Kimi、GLM——多方向范式创新的先发者
- **闭源旗舰系**：GPT、Claude、Gemini——规模与数据领先，定义能力前沿

---

## 9. 批判与反思：这篇文档的局限

按照《研究生学术训练五阶段方法论》中的"批判训练"要求，我尝试对这篇学习笔记提出几个**建设性质疑**：

1. **时间线的准确性**：文档标注为"2026 年 6 月"，但部分模型（如 GPT-5.4、LLaMA 4）的发布时间、参数细节我无法独立验证。技术博客的时效性很强，建议读者以官方技术报告为准。

2. **简化带来的失真**：为了零基础友好，很多概念被大幅简化。例如：
   - MoE 的"专家"并不是真正的领域专家，而是特定 token 模式的处理单元，但文档用"医院分诊"类比可能让人误解为每个专家有明确的专业分工
   - RMSNorm "去掉减均值"的描述在工程实现上比理论描述更复杂

3. **缺少数学推导**：虽然公式被拆解，但真正的理解需要手动推导一次 Attention 矩阵的梯度。纯阅读无法替代笔算。

4. **中国模型的信息来源**：Qwen、DeepSeek、Kimi 等模型的部分描述可能来自官方宣传材料，而非经过同行评审的论文。作为初学者，我暂时没有能力区分"技术突破"和"营销话术"。

5. **缺少代码实践**：这篇博客是纯理论梳理。Transformer 的真正理解必须配合代码——至少手写一次 Self-Attention，用 PyTorch 搭建一次 Encoder-Decoder。下一步我会在另一篇博客中补充代码实现。

> **如果你发现本文有任何事实错误或表述不清的地方，欢迎在评论区指出，我会持续修正。**

---

## 10. 参考文献

1. Vaswani A, Shazeer N, Parmar N, et al. **Attention is all you need**. NeurIPS, 2017.
2. Devlin J, Chang M W, Lee K, et al. **BERT: Pre-training of deep bidirectional transformers for language understanding**. NAACL-HLT, 2019.
3. Brown T, Mann B, Ryder N, et al. **Language models are few-shot learners**. NeurIPS, 2020.
4. Touvron H, Lavril T, Izacard G, et al. **LLaMA: Open and efficient foundation language models**. arXiv, 2023.
5. Ainslie J, Lee-Thorp J, de Jong M, et al. **GQA: Training generalized multi-query transformer models from multi-head checkpoints**. EMNLP, 2023.
6. Shazeer N, Mirhoseini A, Maziarz K, et al. **Outrageously large neural networks: The sparsely-gated mixture-of-experts layer**. ICML, 2017.
7. Ouyang L, Wu J, Jiang X, et al. **Training language models to follow instructions with human feedback**. NeurIPS, 2022.
8. DeepSeek-AI. **DeepSeek-V3 technical report**. arXiv:2412.19437, 2024.
9. DeepSeek-AI. **DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning**. arXiv:2501.12948, 2025.
10. Su J, Lu Y, Pan S, et al. **RoFormer: Enhanced transformer with rotary position embedding**. Neurocomputing, 2024.

---

> **后记**：这篇博客是我从"囤粮食"到"吃粮食"的第一次尝试。如果你也在学习 Transformer，欢迎交换笔记。下一步计划：手写一个最小化的 Transformer（PyTorch 实现），并写一篇配套的代码解读博客。
> 
> **更新日志**：
> - 2026-06-07：初稿完成，基于《Transformer 零基础深入学习指南》整理
