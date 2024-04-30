
# 简介

将该问题匹配任务任务转为自然语言处理中常见的二分类任务，主要选择使用了Ernie-Gram预训练模型。针对该题句对样本之间差距小这一难点，以增大句对样本间差异为切入点，用词性标注模型提取词性特征，引入TOKEN-MASK机制指导模型重点学习两句子的差异部分。我们通过实施一个全面的多策略系统工程对问题匹配任务的模型鲁棒性进行了显著提升。该工程不仅包括数据预处理、模型训练，还涵盖了后处理等多个阶段，涉及了包括词性标注技术、TOKEN-MASK策略、多任务学习、随机对偶增强、FGM对抗训练和TTA等多种策略。通过在严格且富有挑战性的DuQM评测数据集上的实验验证，本多策略系统工程显著提高了模型的整体性能，其得分达到了88.403，相较于先前的强大基线模型得分提升了6.906分。这一成果不仅展示了所采用策略的有效性，也体现了在面对复杂语言扰动时，综合多种策略能够显著增强模型的处理能力。



## 数据预处理： 
    
- 使用百度词性标注模型 分词、词性标注
- 地理位置识别
- 中文、阿拉伯数字统一表示
- 拼音识别
- 细粒度变化识别（插入、替换、位置调换）


## 模型一：ERNIE-Gram单模+对抗训练+数据增强+阈值搜索

传统的BERT + Linear 分类结构，在bert输入时对数据做对偶，前向时不同数据对做两次model，该模型对对话理解项指标有显著优势

## 模型二：ERNIE-Gram+多任务学习+对抗训练+数据增强+阈值搜索
    
针对数据的特点。以编辑距离为划分指标，对于编辑距离大于70的样本，只选择两句中不同的字符token做mean-pooling，学习细粒度知识。对于编辑距离小于70的样本，对所有字符token做mean-pooling、


## 模型后处理

- 同义词纠正：检测出两句子发生替换操作，且不是询问组词、造句、翻译、拼读写，且替换词语为同义词，置为正样本。
- 反义词纠正：检测出两句子发生替换操作，且不是询问组词、造句、翻译、拼读写，且替换词语为反义词，置为负样本。
- 并列关系纠正：检测句子发生交换词语操作，且为并列关系（中间词为表并列关系的词），置为正样本。
- 地名实体交换1: 检测句子发生地名实体的交换，且为交换不改变意思的对称结构，置为正样本。
- 地名实体交换2： 检测句子发生地名实体的交换，且为交换改变意思的结构，置为负样本。
- 主谓替换：检测句子发生事件主体的替换，且交换后意思颠倒（中间词大多为动词），置为负样本。
- 插入词语处理：检测句子发生插入操作，将插入颜色、程度、状态等词语的以及置性度低的样本置0。
- 拼写错误处理：检测句子对只有1个字不一样，且拼音相同置为负样本。（排除询问拼、读写、意思的情况）
- 询问组词类处理：提取需要组词的字主题，比较主题是否相等，相等的置为正样本不相等的置为负样本。
- 其他样本：三套模型方案加权rank融合。


# 代码说明
## 代码结构
```
    angular2html
    |-- B
        |-- Dockerfile
        |-- __init__.py
        |-- docker_build.sh
        |-- infer_lin.sh #
        |-- readme.md
        |-- requirements.txt
        |-- run.sh
        |-- run_lin.sh
        |-- run_xia.sh
        |-- .ipynb_checkpoints
        |-- code1                #模型1和模型1 
        |   |-- __init__.py 
        |   |-- config.yaml         #一些配置文件
        |   |-- config_add_lac.yaml
        |   |-- config_mutitask.yaml
        |   |-- data.py                  #数据读取和数据的转换
        |   |-- dataprepare.py           #数据准备、主要是提取词性和分词
        |   |-- infer_att_cv.py          #模型二 推理
        |   |-- infer_att_lac.py 
        |   |-- infer_multitask.py       #模型二推理
        |   |-- model.py                 #模型文件 所有模型都在这
        |   |-- post2.py                 # 后处理代码
        |   |-- rule.py            
        |   |-- run_att.py               #模型一训练
        |   |-- run_att_lac.py   
        |   |-- run_multitask.py         #模型三训练
        |   |-- runconfig.py
        |   |-- train.py           #训练代码
        |   |-- data_new
        |   |   |-- cuted_testB.csv #dataprepare 生成的测试集数据
        |   |   |-- new_test.csv   #dataprepare 生成的测试集数据
        |   |   |-- new_testB.csv  #dataprepare 生成的测试集数据
        |   |   |-- new_train.csv  #dataprepare 生成的测试集数据
        |   |   |-- test_.csv  
        |   |   |-- 反义词库.txt    #来源于网络 + 训练集统计修正
        |   |   |-- 否定词库.txt    #来源于网络 + 训练集统计修正
        |   |   |-- 新同义词典.txt  #来源于网络 + 训练集统计修正
        |   |-- user_data
        |   |   |-- configs
        |   |   |   |-- attention_fgm_config.pkl
        |   |   |   |-- mutitask_config.pkl
        |   |   |-- models          #存放Ernie方案的模型
        |   |       |-- attention_fgm 
        |   |       |   |-- best_val_stepsingle.pdparams
        |   |       |-- mutitask    
        |   |           |-- best_val_stepsingle.pdparams
        |   |-- utils               
        |       |-- BaseModel.py
        |       |-- __init__.py
        |       |-- attack.py
        |       |-- config.py
        |       |-- dict2Obj.py
        |       |-- log_setting.py
        |       |-- myfile.py
        |       |-- seed.py
        |       |-- threshold.py
           
```



