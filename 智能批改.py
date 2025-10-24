from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate

chatLLM = ChatTongyi(
    model="qwen3-30b-a3b-instruct-2507",
    streaming=False,
    enable_thinking=False,
    max_tokens=50,
)

# 优化后的系统提示词
system_prompt = """你是专业的教育评估助手，专门负责批改学生试卷并提供详细的分析报告。你的核心能力包括：
1. 处理多份学生试卷图像，准确提取题目和答案
2. 根据标准答案进行客观评分
3. 为每位学生提供个性化的错题分析和改进建议
4. 生成班级整体学习情况分析报告

工作原则：
- 严谨细致：确保每份试卷都被准确批改，不遗漏任何题目
- 客观公正：严格按照标准答案评分，不掺杂个人主观判断
- 编号管理：为每份试卷分配唯一编号，确保结果不会混淆
- 结构化输出：按固定格式输出批改结果，便于理解和使用

工作流程：
1. 确认用户提供的信息（标准答案、学生试卷图像数量）
2. 为每份试卷分配编号（按上传顺序或用户指定）
3. 逐份批改试卷并生成个性化分析
4. 汇总所有结果，生成班级整体分析报告"""

# 优化后的用户提示模板
prompt_template = PromptTemplate.from_template(
    """请根据以下信息批改学生试卷并生成详细报告：

标准答案：
{standard_answers}

学生试卷图像：
共 {student_count} 份{image_info}

请按以下格式输出批改结果：

# 批改概况
- 总试卷数量：{student_count}份
- 可批改数量：X份（学生1-X）
- 不可批改数量：Y份（说明原因）

---

# 逐份批改结果

## 学生1
### 得分
**XX/100**

### 错题分析
1. 第X题
   - 题目：[题目内容]
   - 学生答案：[学生答案]
   - 错误原因：[具体分析]
   - 改进建议：[针对性建议]

### 针对性练习
1. [练习题1]
2. [练习题2]
3. [练习题3]

（后续学生按相同格式）

---

# 整体分析报告

### 核心数据
- 平均分：XX分
- 分数段分布：[各分数段人数]
- 及格率：XX%
- 高分率：XX%

### 共性问题
- 高频错题：[错题及错误率]
- 知识点薄弱环节：[具体知识点]
- 常见错误类型：[错误类型分析]

### 教学建议
- 重点强化内容：[建议]
- 分层教学策略：[针对不同水平学生的建议]

### 班级练习题
1. [练习题1]
2. [练习题2]
3. [练习题3]"""
)

# 使用示例
standard_answers = """
1. 题目：解一元二次方程 x²-5x+6=0
   答案：x=2或x=3
   评分标准：公式正确得2分，计算正确得2分，答案正确得1分

2. 题目：简述光合作用的过程
   答案：光合作用是指绿色植物在光照条件下，利用二氧化碳和水合成有机物并释放氧气的过程。
   评分标准：关键词完整得3分，表述清晰得2分
"""

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=prompt_template.format(
        standard_answers=standard_answers,
        student_count=3,
        image_info="（按上传顺序编号）"
    ))
]

# 执行批改
res = chatLLM.invoke(messages)
print("批改结果:")
print(res.content)