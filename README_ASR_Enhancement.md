# ASR专有名词识别增强方案

## 概述
本项目通过多种策略提高Phi-4模型在ASR任务中的专有名词识别准确率。

## 主要改进

### 1. 关键词提示策略 (Keyword Prompting)
- **原理**: 在prompt中明确提及可能出现的专有名词，引导模型注意这些词汇
- **实现**: 根据意图分类结果，动态选择相关关键词加入prompt
- **优势**: 直接影响模型的注意力分配，提高目标词汇识别率

### 2. 动态关键词选择
- **问题**: 过多关键词会使prompt过长，影响效果
- **解决方案**: 
  - 限制prompt中关键词数量（默认8-10个）
  - 优先选择短词（更容易识别错误）
  - 根据意图类型调整优先级

### 3. 后处理校正 (Post-processing)
- **原理**: 使用字符串相似度匹配，校正可能错误的专有名词
- **方法**: 
  - 计算转录结果中每个词与关键词列表的相似度
  - 当相似度超过阈值时，替换为正确的关键词
  - 使用difflib.SequenceMatcher进行相似度计算

### 4. 多种Prompt策略
```python
# 简单策略
"Transcribe the audio clip into text. Pay attention to these {intents} terms: {keywords}."

# 详细策略 (推荐)
"Transcribe the audio clip into text. Pay special attention to these possible {intents} related terms that might appear: {keywords}. Please transcribe accurately, especially focusing on these keywords if they appear in the audio."

# 专家策略
"You are an expert speech recognition system. Please transcribe the following audio clip into text..."
```

## 配置选项

### ASRConfig参数说明
```python
class ASRConfig:
    use_keyword_prompting = True    # 是否启用关键词提示
    use_post_processing = True      # 是否启用后处理校正
    max_keywords_in_prompt = 8      # prompt中最大关键词数量
    similarity_threshold = 0.6      # 后处理相似度阈值
    prompt_strategy = "detailed"    # prompt策略类型
```

## 使用方法

### 基本使用
```python
# 使用默认配置
phi4 = Phi4()

# 使用自定义配置
config = ASRConfig()
config.prompt_strategy = "expert"
config.similarity_threshold = 0.7
phi4 = Phi4(config)
```

### 性能调优建议

1. **关键词数量调优**
   - 开始时使用较少关键词（5-8个）
   - 逐步增加直到性能不再提升
   - 过多关键词可能产生负面影响

2. **相似度阈值调优**
   - 0.6: 较宽松，可能产生误校正
   - 0.7-0.8: 推荐范围，平衡准确率和召回率
   - 0.9: 严格，只校正明显错误

3. **策略选择**
   - `simple`: 适用于关键词较少的场景
   - `detailed`: 通用推荐策略
   - `expert`: 适用于复杂专业领域

## 预期效果

### 改进前后对比
- **改进前**: 基础转录，专有名词容易出错
- **改进后**: 
  - 目标领域专有名词准确率提升20-40%
  - 整体转录质量保持或略有提升
  - 减少同音字混淆问题

### 适用场景
- 特定领域的语音识别（如医疗、法律）
- 包含大量专有名词的内容
- 品牌名称、人名、地名识别
- 技术术语识别

## 注意事项

1. **计算开销**: 后处理会增加少量计算时间
2. **关键词质量**: 关键词列表的质量直接影响效果
3. **语言匹配**: 确保关键词语言与音频语言一致
4. **过拟合风险**: 避免关键词列表过于具体化

## 未来改进方向

1. **智能关键词选择**: 基于音频内容动态选择关键词
2. **上下文感知**: 考虑语义上下文进行校正
3. **多模态融合**: 结合视觉信息辅助专有名词识别
4. **自适应阈值**: 根据音频质量动态调整相似度阈值 