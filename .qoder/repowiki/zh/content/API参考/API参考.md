# API参考

<cite>
**本文档引用的文件**
- [qlib/__init__.py](file://qlib/__init__.py)
- [qlib/data/__init__.py](file://qlib/data/__init__.py)
- [qlib/model/__init__.py](file://qlib/model/__init__.py)
- [qlib/backtest/__init__.py](file://qlib/backtest/__init__.py)
- [qlib/workflow/__init__.py](file://qlib/workflow/__init__.py)
- [qlib/data/data.py](file://qlib/data/data.py)
- [qlib/model/base.py](file://qlib/model/base.py)
- [qlib/workflow/exp.py](file://qlib/workflow/exp.py)
- [qlib/workflow/recorder.py](file://qlib/workflow/recorder.py)
- [qlib/config.py](file://qlib/config.py)
- [qlib/data/cache.py](file://qlib/data/cache.py)
- [qlib/data/dataset/__init__.py](file://qlib/data/dataset/__init__.py)
- [qlib/workflow/expm.py](file://qlib/workflow/expm.py)
</cite>

## 目录
1. [引言](#引言)
2. [初始化模块](#初始化模块)
3. [数据模块](#数据模块)
4. [模型模块](#模型模块)
5. [回测模块](#回测模块)
6. [工作流模块](#工作流模块)
7. [版本兼容性与API演进](#版本兼容性与api演进)

## 引言
Qlib是一个用于量化投资研究的机器学习框架，提供了一套完整的API来支持从数据获取、模型训练到回测和投资组合管理的全流程。本API参考文档详细介绍了Qlib的核心公共接口，包括数据、模型、回测和工作流等主要模块。文档基于代码中的docstring自动生成，并补充了实际使用示例，旨在为开发者提供全面的技术参考。

## 初始化模块

### qlib.init()
初始化Qlib框架，配置全局参数和组件。

**参数**
- `default_conf` (str): 默认配置模式，可选值为"client"或"server"。默认值为"client"。
- `clear_mem_cache` (bool): 是否清除内存缓存。默认值为True。
- `skip_if_reg` (bool): 当使用记录器时，设置为True可避免记录器丢失。默认值为False。

**返回值**
- 无

**异常情况**
- 无

**使用示例**
```python
from qlib import init
init(default_conf="client", clear_mem_cache=True)
```

**Section sources**
- [qlib/__init__.py](file://qlib/__init__.py#L25-L85)

### qlib.auto_init()
自动初始化Qlib框架，按照优先级顺序查找项目配置并初始化。

**参数**
- `cur_path` (Union[Path, str, None]): 查找项目路径的起始路径。

**返回值**
- 无

**异常情况**
- `FileNotFoundError`: 如果找不到项目路径。

**使用示例**
```python
from qlib import auto_init
auto_init()
```

**Section sources**
- [qlib/__init__.py](file://qlib/__init__.py#L243-L318)

## 数据模块

### qlib.data.D
数据访问接口，提供对日历、标的、特征、表达式和数据集的访问。

**属性**
- `D`: 数据访问对象，通过该对象可以访问各种数据提供者。

**使用示例**
```python
from qlib.data import D
instruments = D.instruments("csi500")
```

**Section sources**
- [qlib/data/__init__.py](file://qlib/data/__init__.py#L8-L27)

### D.instruments()
获取标的列表或配置。

**参数**
- `market` (Union[List, str]): 市场或标的列表。可以是字符串（如"all", "csi500"）或标的代码列表。
- `filter_pipe` (Union[List, None]): 动态过滤器列表。

**返回值**
- Union[dict, list]: 如果`market`是字符串，则返回标的配置字典；如果`market`是列表，则直接返回该列表。

**异常情况**
- 无

**使用示例**
```python
from qlib.data import D
# 获取CSI500指数成分股
instruments = D.instruments("csi500")
# 获取特定股票列表
instruments = D.instruments(["SH600000", "SZ000001"])
```

**Section sources**
- [qlib/data/data.py](file://qlib/data/data.py#L206-L265)

### D.calendar()
获取指定市场在给定时间范围内的交易日历。

**参数**
- `start_time` (str): 时间范围的开始。
- `end_time` (str): 时间范围的结束。
- `freq` (str): 时间频率，可选值为year/quarter/month/week/day。
- `future` (bool): 是否包含未来交易日。

**返回值**
- list: 交易日历列表。

**异常情况**
- 无

**使用示例**
```python
from qlib.data import D
calendar = D.calendar(start_time="2020-01-01", end_time="2020-12-31", freq="day")
```

**Section sources**
- [qlib/data/data.py](file://qlib/data/data.py#L71-L110)

### D.feature()
获取特定标的的特征数据。

**参数**
- `instrument` (str): 标的代码。
- `field` (str): 特征字段。
- `start_time` (str): 时间范围的开始。
- `end_time` (str): 时间范围的结束。
- `freq` (str): 时间频率。

**返回值**
- pd.Series: 特征数据序列。

**异常情况**
- 无

**使用示例**
```python
from qlib.data import D
feature = D.feature("SH600000", "$close", "2020-01-01", "2020-12-31", "day")
```

**Section sources**
- [qlib/data/data.py](file://qlib/data/data.py#L314-L335)

### D.expression()
获取表达式数据。

**参数**
- `instrument` (str): 标的代码。
- `field` (str): 表达式字段。
- `start_time` (str): 时间范围的开始。
- `end_time` (str): 时间范围的结束。
- `freq` (str): 时间频率。

**返回值**
- pd.Series: 表达式数据序列。

**异常情况**
- 无

**使用示例**
```python
from qlib.data import D
expression = D.expression("SH600000", "Ref($close, -1)/$close-1", "2020-01-01", "2020-12-31", "day")
```

**Section sources**
- [qlib/data/data.py](file://qlib/data/data.py#L410-L443)

### D.dataset()
获取数据集数据。

**参数**
- `instruments` (list or dict): 标的列表或字典，或标的池配置字典。
- `fields` (list): 特征实例列表。
- `start_time` (str): 时间范围的开始。
- `end_time` (str): 时间范围的结束。
- `freq` (str): 时间频率。
- `inst_processors` (Iterable[Union[dict, InstProcessor]]): 对每个标的执行的操作。

**返回值**
- pd.DataFrame: 具有<标的, 日期>索引的pandas数据框。

**异常情况**
- 无

**使用示例**
```python
from qlib.data import D
fields = ["$open", "$close", "$high", "$low", "$volume"]
data = D.dataset("csi500", fields, "2020-01-01", "2020-12-31", "day")
```

**Section sources**
- [qlib/data/data.py](file://qlib/data/data.py#L453-L476)

## 模型模块

### qlib.model.Model
模型基类，定义了模型训练和预测的接口。

**方法**
- `fit(dataset: Dataset, reweighter: Reweighter)`: 从数据集中学习模型。
- `predict(dataset: Dataset, segment: Union[Text, slice] = "test") -> object`: 给定数据集进行预测。

**参数**
- `dataset` (Dataset): 用于模型训练的处理后数据。
- `reweighter` (Reweighter): 重加权器。
- `segment` (Text or slice): 数据集用于准备数据的片段。默认为"test"。

**返回值**
- object: 预测结果，如`pandas.Series`。

**异常情况**
- `NotImplementedError`: 子类必须实现这些方法。

**使用示例**
```python
from qlib.model.base import Model
class MyModel(Model):
    def fit(self, dataset, reweighter):
        # 实现模型训练逻辑
        pass
    
    def predict(self, dataset, segment="test"):
        # 实现预测逻辑
        pass
```

**Section sources**
- [qlib/model/base.py](file://qlib/model/base.py#L25-L78)

### qlib.model.ModelFT
可微调模型基类，支持模型微调。

**方法**
- `finetune(dataset: Dataset)`: 基于给定数据集微调模型。

**参数**
- `dataset` (Dataset): 用于模型训练的处理后数据。

**返回值**
- 无

**异常情况**
- `NotImplementedError`: 子类必须实现此方法。

**使用示例**
```python
from qlib.model.base import ModelFT
class MyModelFT(ModelFT):
    def finetune(self, dataset):
        # 实现模型微调逻辑
        pass
```

**Section sources**
- [qlib/model/base.py](file://qlib/model/base.py#L84-L109)

## 回测模块

### qlib.backtest.backtest()
执行回测，初始化策略和执行器，并进行交互。

**参数**
- `start_time` (Union[pd.Timestamp, str]): 回测的闭合开始时间。
- `end_time` (Union[pd.Timestamp, str]): 回测的闭合结束时间。
- `strategy` (Union[str, dict, object, Path]): 用于初始化最外层投资组合策略的配置。
- `executor` (Union[str, dict, object, Path]): 用于初始化最外层执行器的配置。
- `benchmark` (str): 用于报告的基准。
- `account` (Union[float, int, dict]): 描述如何创建账户的信息。
- `exchange_kwargs` (dict): 用于初始化交易所的kwargs。
- `pos_type` (str): 位置类型。

**返回值**
- Tuple[PORT_METRIC, INDICATOR_METRIC]: 交易投资组合指标信息和交易指标。

**异常情况**
- 无

**使用示例**
```python
from qlib.backtest import backtest
result = backtest(
    start_time="2020-01-01",
    end_time="2020-12-31",
    strategy=strategy_config,
    executor=executor_config,
    benchmark="SH000300"
)
```

**Section sources**
- [qlib/backtest/__init__.py](file://qlib/backtest/__init__.py#L217-L276)

### qlib.backtest.get_exchange()
获取交易所实例。

**参数**
- `exchange` (Union[str, dict, object, Path]): 交易所配置。
- `freq` (str): 数据频率。
- `start_time` (Union[pd.Timestamp, str]): 回测的闭合开始时间。
- `end_time` (Union[pd.Timestamp, str]): 回测的闭合结束时间。
- `codes` (Union[list, str]): 标的代码列表或字符串（如"all", "csi500"）。
- `subscribe_fields` (list): 订阅字段。
- `open_cost` (float): 开仓交易成本比例。
- `close_cost` (float): 平仓交易成本比例。
- `min_cost` (float): 最小交易成本。
- `limit_threshold` (Union[Tuple[str, str], float, None]): 涨跌停限制。
- `deal_price` (Union[str, Tuple[str, str], List[str]]): 成交价格。

**返回值**
- Exchange: 初始化的交易所对象。

**异常情况**
- 无

**使用示例**
```python
from qlib.backtest import get_exchange
exchange = get_exchange(
    freq="day",
    start_time="2020-01-01",
    end_time="2020-12-31",
    codes="csi500"
)
```

**Section sources**
- [qlib/backtest/__init__.py](file://qlib/backtest/__init__.py#L33-L110)

## 工作流模块

### qlib.workflow.R
全局工作流记录器，用于管理实验和记录。

**属性**
- `R`: 全局记录器实例。

**方法**
- `start()`: 开始一个实验。
- `start_exp()`: 开始一个实验（低级方法）。
- `end_exp()`: 手动结束一个实验。
- `search_records()`: 搜索符合标准的记录。
- `list_experiments()`: 列出所有现有实验。
- `list_recorders()`: 列出实验的所有记录器。
- `get_exp()`: 获取实验实例。
- `delete_exp()`: 删除实验。
- `get_uri()`: 获取当前实验管理器的URI。
- `set_uri()`: 重置当前实验管理器的默认URI。
- `uri_context()`: 临时设置实验管理器的默认URI。
- `get_recorder()`: 获取记录器实例。
- `delete_recorder()`: 删除记录器。
- `save_objects()`: 保存对象作为实验的工件。
- `load_object()`: 从实验的工件中加载对象。
- `log_params()`: 记录参数。
- `log_metrics()`: 记录指标。
- `log_artifact()`: 记录工件。
- `download_artifact()`: 下载工件。
- `set_tags()`: 设置标签。

**使用示例**
```python
from qlib.workflow import R
with R.start(experiment_name='test', recorder_name='recorder_1'):
    model.fit(dataset)
    R.log_metrics(train_loss=0.33, step=1)
```

**Section sources**
- [qlib/workflow/__init__.py](file://qlib/workflow/__init__.py#L26-L681)

### R.start_exp()
开始一个实验的低级方法。

**参数**
- `experiment_id` (str): 要启动的实验ID。
- `experiment_name` (str): 要启动的实验名称。
- `recorder_id` (str): 要启动的记录器ID。
- `recorder_name` (str): 要启动的记录器名称。
- `uri` (str): 实验的跟踪URI。
- `resume` (bool): 是否恢复特定记录器。

**返回值**
- Experiment: 正在启动的实验实例。

**异常情况**
- 无

**使用示例**
```python
from qlib.workflow import R
R.start_exp(experiment_name='test', recorder_name='recorder_1')
# 进一步操作
R.end_exp('FINISHED')
```

**Section sources**
- [qlib/workflow/__init__.py](file://qlib/workflow/__init__.py#L97-L145)

### R.get_exp()
获取实验实例。

**参数**
- `experiment_id` (str): 实验ID。
- `experiment_name` (str): 实验名称。
- `create` (bool): 如果实验不存在，是否自动创建。
- `start` (bool): 如果创建了新实验，是否启动。

**返回值**
- Experiment: 具有给定ID或名称的实验实例。

**异常情况**
- 无

**使用示例**
```python
from qlib.workflow import R
exp = R.get_exp(experiment_name='test')
recorders = exp.list_recorders()
```

**Section sources**
- [qlib/workflow/__init__.py](file://qlib/workflow/__init__.py#L243-L323)

## 版本兼容性与API演进

### 版本兼容性策略
Qlib遵循语义化版本控制（Semantic Versioning）原则，确保API的向后兼容性。主要版本号的变更表示不兼容的API更改，次要版本号的变更表示向后兼容的功能添加，补丁版本号的变更表示向后兼容的问题修复。

### API演进路线
Qlib的API演进将遵循以下原则：
1. **稳定性优先**：核心API将保持稳定，避免频繁变更。
2. **渐进式改进**：新功能将通过新增接口的方式引入，而不是修改现有接口。
3. **向后兼容**：在可能的情况下，保持对旧版本API的兼容性，通过弃用警告逐步引导用户迁移到新API。
4. **文档完善**：每次API变更都将伴随详细的文档更新，包括变更原因、影响范围和迁移指南。

**Section sources**
- [qlib/__init__.py](file://qlib/__init__.py#L5-L11)
- [qlib/config.py](file://qlib/config.py#L13-L25)