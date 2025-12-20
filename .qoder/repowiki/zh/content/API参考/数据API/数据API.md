# 数据API

<cite>
**本文档引用的文件**   
- [__init__.py](file://qlib/data/__init__.py)
- [data.py](file://qlib/data/data.py)
- [base.py](file://qlib/data/base.py)
- [cache.py](file://qlib/data/cache.py)
- [ops.py](file://qlib/data/ops.py)
- [file_storage.py](file://qlib/data/storage/file_storage.py)
- [high_performance_ds.py](file://qlib/backtest/high_performance_ds.py)
- [data_cache_demo.py](file://examples/data_demo/data_cache_demo.py)
- [data_mem_resuse_demo.py](file://examples/data_demo/data_mem_resuse_demo.py)
- [dataset/__init__.py](file://qlib/data/dataset/__init__.py)
- [loader.py](file://qlib/data/dataset/loader.py)
</cite>

## 目录
1. [数据提供者](#数据提供者)
2. [数据集](#数据集)
3. [处理器](#处理器)
4. [操作符](#操作符)
5. [缓存机制](#缓存机制)
6. [核心函数详解](#核心函数详解)
7. [数据存储接口扩展](#数据存储接口扩展)
8. [高性能数据结构](#高性能数据结构)

## 数据提供者

qlib.data模块的核心是数据提供者（DataProvider）体系，它通过一系列抽象基类和具体实现来提供不同类型的数据。主要的数据提供者包括CalendarProvider（日历）、InstrumentProvider（标的）、FeatureProvider（特征）和ExpressionProvider（表达式）。

这些提供者遵循统一的设计模式：定义抽象接口，由具体实现类（如LocalCalendarProvider）负责从本地或远程加载数据。这种设计实现了数据访问逻辑与数据源的解耦，使得系统可以灵活地支持多种数据源。

```mermaid
classDiagram
class ProviderBackendMixin {
+get_default_backend() dict
+backend_obj(**kwargs) object
}
class CalendarProvider {
<<abstract>>
+calendar(start_time, end_time, freq, future) list
+locate_index(start_time, end_time, freq, future) tuple
_get_calendar(freq, future) tuple
load_calendar(freq, future) list
}
class InstrumentProvider {
<<abstract>>
+instruments(market, filter_pipe) dict or list
+list_instruments(instruments, start_time, end_time, freq, as_list) dict or list
get_inst_type(inst) str
}
class FeatureProvider {
<<abstract>>
+feature(instrument, field, start_time, end_time, freq) pd.Series
}
class ExpressionProvider {
<<abstract>>
+expression(instrument, field, start_time, end_time, freq) pd.Series
get_expression_instance(field) Expression
}
class DatasetProvider {
<<abstract>>
+dataset(instruments, fields, start_time, end_time, freq, inst_processors) pd.DataFrame
dataset_processor(instruments_d, column_names, start_time, end_time, freq, inst_processors) pd.DataFrame
inst_calculator(inst, start_time, end_time, freq, column_names, spans, g_config, inst_processors) pd.DataFrame
}
class LocalCalendarProvider {
+load_calendar(freq, future) list
}
class LocalInstrumentProvider {
+_load_instruments(market, freq) dict
+list_instruments(instruments, start_time, end_time, freq, as_list) dict or list
}
class LocalFeatureProvider {
+feature(instrument, field, start_index, end_index, freq) pd.Series
}
class LocalExpressionProvider {
+expression(instrument, field, start_time, end_time, freq) pd.Series
}
class LocalDatasetProvider {
+dataset(instruments, fields, start_time, end_time, freq, inst_processors) pd.DataFrame
}
ProviderBackendMixin <|-- CalendarProvider
ProviderBackendMixin <|-- InstrumentProvider
ProviderBackendMixin <|-- FeatureProvider
ProviderBackendMixin <|-- DatasetProvider
CalendarProvider <|-- LocalCalendarProvider
InstrumentProvider <|-- LocalInstrumentProvider
FeatureProvider <|-- LocalFeatureProvider
ExpressionProvider <|-- LocalExpressionProvider
DatasetProvider <|-- LocalDatasetProvider
```

**图源**
- [data.py](file://qlib/data/data.py#L65-L295)
- [data.py](file://qlib/data/data.py#L637-L724)

**节源**
- [data.py](file://qlib/data/data.py#L65-L295)
- [data.py](file://qlib/data/data.py#L637-L724)

## 数据集

数据集（Dataset）是qlib中用于模型训练和推理的数据准备组件。它负责将原始数据转换为模型所需的格式。DatasetH是核心实现，它结合了DataHandler来处理数据预处理逻辑。

数据集的设计允许用户将数据预处理逻辑（如特征工程、标准化）放在DataHandler中，而数据集本身主要负责数据分割和特定于模型的处理。这种分离使得数据处理逻辑更加模块化和可重用。

```mermaid
classDiagram
class Dataset {
<<Serializable>>
+__init__(**kwargs) void
+config(**kwargs) void
+setup_data(**kwargs) void
+prepare(**kwargs) object
}
class DatasetH {
+__init__(handler, segments, fetch_kwargs, **kwargs) void
+config(handler_kwargs, **kwargs) void
+setup_data(handler_kwargs, **kwargs) void
_prepare_seg(slc, **kwargs) object
+prepare(segments, col_set, data_key, **kwargs) Union[List[pd.DataFrame], pd.DataFrame]
get_min_time(segments) datetime
get_max_time(segments) datetime
}
class TSDatasetH {
+__init__(step_len, flt_col, **kwargs) void
+config(**kwargs) void
+setup_data(**kwargs) void
_prepare_seg(slc, **kwargs) TSDataSampler
_extend_slice(slc, cal, step_len) slice
}
class TSDataSampler {
+__init__(data, start, end, step_len, fillna_type, dtype, flt_data) void
+get_index() pd.MultiIndex
+config(**kwargs) void
build_index(data) Tuple[pd.DataFrame, dict]
_get_indices(row, col) np.array
_get_row_col(idx) Tuple[int]
+__getitem__(idx) np.ndarray
+__len__() int
+empty bool
}
Dataset <|-- DatasetH
DatasetH <|-- TSDatasetH
TSDatasetH --> TSDataSampler
```

**图源**
- [dataset/__init__.py](file://qlib/data/dataset/__init__.py#L15-L722)

**节源**
- [dataset/__init__.py](file://qlib/data/dataset/__init__.py#L15-L722)

## 处理器

处理器（Processor）是数据预处理的核心组件，通常作为DataHandler的一部分使用。它们负责执行各种数据转换操作，如缺失值填充、标准化、去极值等。处理器的设计遵循链式调用模式，可以将多个处理器串联起来形成复杂的数据处理流水线。

在DatasetProvider中，inst_processors参数允许在获取数据时应用处理器，这为数据处理提供了极大的灵活性。

```mermaid
classDiagram
class InstProcessor {
<<abstract>>
+__call__(data, instrument) pd.DataFrame
}
class DropnaProcessor {
+__call__(data, instrument) pd.DataFrame
}
class FillnaProcessor {
+__call__(data, instrument) pd.DataFrame
}
class RobustZScoreNormProcessor {
+__call__(data, instrument) pd.DataFrame
+fit(data, instrument) void
+transform(data, instrument) pd.DataFrame
}
class CSRankNormProcessor {
+__call__(data, instrument) pd.DataFrame
}
InstProcessor <|-- DropnaProcessor
InstProcessor <|-- FillnaProcessor
InstProcessor <|-- RobustZScoreNormProcessor
InstProcessor <|-- CSRankNormProcessor
```

**图源**
- [dataset/processor.py](file://qlib/data/dataset/processor.py)

**节源**
- [dataset/processor.py](file://qlib/data/dataset/processor.py)

## 操作符

操作符（Ops）是qlib中用于构建特征表达式的核心组件。它们基于表达式（Expression）基类实现，支持各种数学运算、逻辑运算和时间序列运算。操作符的设计使得用户可以用类似公式的方式定义复杂的特征。

操作符分为几大类：元素级运算符（ElemOperator）、成对运算符（PairOperator）、滚动运算符（Rolling）等。这种分层设计使得操作符的扩展非常方便。

```mermaid
classDiagram
class Expression {
<<abstract>>
+__str__() str
+__repr__() str
+load(instrument, start_index, end_index, *args) pd.Series
_load_internal(instrument, start_index, end_index, *args) pd.Series
get_longest_back_rolling() int
get_extended_window_size() tuple
}
class ExpressionOps {
<<abstract>>
}
class ElemOperator {
<<abstract>>
+__init__(feature) void
+__str__() str
get_longest_back_rolling() int
get_extended_window_size() tuple
}
class NpElemOperator {
+__init__(feature, func) void
_load_internal(instrument, start_index, end_index, *args) pd.Series
}
class PairOperator {
<<abstract>>
+__init__(feature_left, feature_right) void
+__str__() str
get_longest_back_rolling() int
get_extended_window_size() tuple
}
class NpPairOperator {
+__init__(feature_left, feature_right, func) void
_load_internal(instrument, start_index, end_index, *args) pd.Series
}
class Rolling {
+__init__(feature, N, func) void
+__str__() str
_load_internal(instrument, start_index, end_index, *args) pd.Series
get_longest_back_rolling() int
get_extended_window_size() tuple
}
Expression <|-- ExpressionOps
ExpressionOps <|-- ElemOperator
ExpressionOps <|-- PairOperator
ExpressionOps <|-- Rolling
ElemOperator <|-- NpElemOperator
PairOperator <|-- NpPairOperator
```

**图源**
- [base.py](file://qlib/data/base.py#L13-L281)
- [ops.py](file://qlib/data/ops.py#L37-L780)

**节源**
- [base.py](file://qlib/data/base.py#L13-L281)
- [ops.py](file://qlib/data/ops.py#L37-L780)

## 缓存机制

qlib的缓存机制是其高性能的关键。它实现了多级缓存策略，包括内存缓存（MemCache）和磁盘缓存（DiskCache）。内存缓存基于OrderedDict实现，支持按长度或大小限制的LRU淘汰策略。

磁盘缓存主要用于缓存计算密集型的特征数据集，避免重复计算。DiskDatasetCache和DiskExpressionCache是主要的磁盘缓存实现，它们使用HDF5格式存储数据，并通过Redis锁保证多进程环境下的读写安全。

```mermaid
classDiagram
class MemCacheUnit {
<<abstract>>
+__init__(*args, **kwargs) void
+__setitem__(key, value) void
+__getitem__(key) object
+__contains__(key) bool
+__len__() int
+set_limit_size(limit) void
+clear() void
+popitem(last) tuple
+pop(key) object
_adjust_size(key, value) void
_get_value_size(value) int
}
class MemCacheLengthUnit {
+_get_value_size(value) int
}
class MemCacheSizeofUnit {
+_get_value_size(value) int
}
class MemCache {
+__init__(mem_cache_size_limit, limit_type) void
+__getitem__(key) MemCacheUnit
+clear() void
}
class BaseProviderCache {
+__init__(provider) void
+__getattr__(attr) object
check_cache_exists(cache_path, suffix_list) bool
clear_cache(cache_path) void
get_cache_dir(dir_name, freq) Path
}
class ExpressionCache {
<<abstract>>
+expression(instrument, field, start_time, end_time, freq) pd.Series
_uri(instrument, field, start_time, end_time, freq) str
_expression(instrument, field, start_time, end_time, freq) pd.Series
update(cache_uri, freq) int
}
class DatasetCache {
<<abstract>>
+dataset(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors) pd.DataFrame
_uri(instruments, fields, start_time, end_time, freq, **kwargs) str
_dataset(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors) pd.DataFrame
_dataset_uri(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors) str
update(cache_uri, freq) int
cache_to_origin_data(data, fields) pd.DataFrame
normalize_uri_args(instruments, fields, freq) tuple
}
class DiskExpressionCache {
+__init__(provider, **kwargs) void
get_cache_dir(freq) Path
_uri(instrument, field, start_time, end_time, freq) str
_expression(instrument, field, start_time, end_time, freq) pd.Series
gen_expression_cache(expression_data, cache_path, instrument, field, freq, last_update) void
update(sid, cache_uri, freq) int
}
class DiskDatasetCache {
+__init__(provider, **kwargs) void
_uri(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors, **kwargs) str
get_cache_dir(freq) Path
read_data_from_cache(cache_path, start_time, end_time, fields) pd.DataFrame
_dataset(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors) pd.DataFrame
_dataset_uri(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors) str
}
MemCacheUnit <|-- MemCacheLengthUnit
MemCacheUnit <|-- MemCacheSizeofUnit
BaseProviderCache <|-- ExpressionCache
BaseProviderCache <|-- DatasetCache
ExpressionCache <|-- DiskExpressionCache
DatasetCache <|-- DiskDatasetCache
```

**图源**
- [cache.py](file://qlib/data/cache.py#L43-L800)

**节源**
- [cache.py](file://qlib/data/cache.py#L43-L800)

## 核心函数详解

### D.instruments()

`D.instruments()`函数用于获取标的池配置。它接受market参数指定市场范围（如"all"、"csi300"等），并可选地通过filter_pipe参数添加动态过滤器。该函数返回一个字典，包含市场名称和过滤器配置，或直接返回标的列表。

```python
# 获取CSI300指数成分股
instruments = D.instruments("csi300")

# 添加动态过滤器
filter_pipe = [
    {"filter_type": "ExpressionDFilter", "rule_expression": "$open<40"},
    {"filter_type": "NameDFilter", "name_rule_re": "SH[0-9]{4}55"}
]
instruments = D.instruments("csi500", filter_pipe=filter_pipe)
```

**节源**
- [data.py](file://qlib/data/data.py#L206-L264)

### D.features()

`D.features()`函数是获取特征数据的核心接口。它接受标的列表、特征字段列表和时间范围，返回一个以<标的, 时间>为索引的DataFrame。该函数内部使用DatasetProvider和缓存机制来高效地获取数据。

```python
# 获取多个标的的多个特征
instruments = ["SH600000", "SH600004"]
fields = ["$close", "$volume", "Ref($close, 1)"]
data = D.features(instruments, fields, start_time="2020-01-01", end_time="2020-12-31")
```

**节源**
- [data.py](file://qlib/data/data.py#L453-L634)

### D.calendar()

`D.calendar()`函数用于获取指定时间范围内的交易日历。它支持多种频率（年、季、月、周、日）和未来交易日选项。该函数利用内存缓存来提高性能，避免重复加载日历数据。

```python
# 获取2020年的日线交易日历
calendar = D.calendar(start_time="2020-01-01", end_time="2020-12-31", freq="day")

# 获取包含未来交易日的日历
future_calendar = D.calendar(freq="day", future=True)
```

**节源**
- [data.py](file://qlib/data/data.py#L71-L109)

## 数据存储接口扩展

qlib通过storage模块提供了可扩展的数据存储接口。FileStorage是主要的实现，它定义了日历、标的和特征数据的存储方式。用户可以通过实现自定义的Storage类来支持新的数据存储后端。

```mermaid
classDiagram
class CalendarStorage {
<<abstract>>
+data list
+check() void
+clear() void
+index(value) int
+insert(index, value) void
+remove(value) void
+__setitem__(i, values) void
+__delitem__(i) void
+__getitem__(i) Union[CalVT, List[CalVT]]
+__len__() int
}
class InstrumentStorage {
<<abstract>>
+data dict
+check() void
+clear() void
+__setitem__(k, v) void
+__delitem__(k) void
+__getitem__(k) InstVT
+update(*args, **kwargs) void
+__len__() int
}
class FeatureStorage {
<<abstract>>
+data pd.Series
+clear() void
+write(data_array, index) void
+start_index int
+end_index int
+__getitem__(i) Union[Tuple[int, float], pd.Series]
+__len__() int
}
class FileStorageMixin {
+provider_uri Path
+dpm DataPathManager
+support_freq List[str]
+uri Path
+check() void
}
class FileCalendarStorage {
+__init__(freq, future, provider_uri, **kwargs) void
+_read_calendar() List[CalVT]
+_write_calendar(values, mode) void
+data list
+extend(values) void
}
class FileInstrumentStorage {
+__init__(market, freq, provider_uri, **kwargs) void
+_read_instrument() Dict[InstKT, InstVT]
+_write_instrument(data) void
+data dict
}
class FileFeatureStorage {
+__init__(instrument, field, freq, provider_uri, **kwargs) void
+clear() void
+data pd.Series
+write(data_array, index) void
+start_index int
+end_index int
+__getitem__(i) Union[Tuple[int, float], pd.Series]
+__len__() int
}
FileStorageMixin <|-- FileCalendarStorage
FileStorageMixin <|-- FileInstrumentStorage
FileStorageMixin <|-- FileFeatureStorage
CalendarStorage <|-- FileCalendarStorage
InstrumentStorage <|-- FileInstrumentStorage
FeatureStorage <|-- FileFeatureStorage
```

**图源**
- [file_storage.py](file://qlib/data/storage/file_storage.py)

**节源**
- [file_storage.py](file://qlib/data/storage/file_storage.py)

## 高性能数据结构

qlib在backtest模块中提供了高性能数据结构，用于优化回测性能。NumpyQuote和PandasQuote是两种主要的报价数据结构，它们针对不同的使用场景进行了优化。

NumpyQuote使用numpy数组存储数据，适合需要高性能数值计算的场景；PandasQuote则保持了pandas DataFrame的灵活性，适合需要复杂数据操作的场景。

```mermaid
classDiagram
class BaseQuote {
<<abstract>>
+__init__(quote_df, freq) void
+get_all_stock() Iterable
+get_data(stock_id, start_time, end_time, field, method) Union[None, int, float, bool, IndexData]
}
class PandasQuote {
+__init__(quote_df, freq) void
+get_all_stock() Iterable
+get_data(stock_id, start_time, end_time, field, method) Union[None, int, float, bool, IndexData]
}
class NumpyQuote {
+__init__(quote_df, freq, region) void
+get_all_stock() Iterable
+get_data(stock_id, start_time, end_time, field, method) Union[None, int, float, bool, IndexData]
_agg_data(data, method) Union[IndexData, np.ndarray, None]
}
class BaseSingleMetric {
<<abstract>>
+__init__(metric) void
+__add__(other) BaseSingleMetric
+__radd__(other) BaseSingleMetric
+__sub__(other) BaseSingleMetric
+__rsub__(other) BaseSingleMetric
+__mul__(other) BaseSingleMetric
+__truediv__(other) BaseSingleMetric
+__eq__(other) BaseSingleMetric
+__gt__(other) BaseSingleMetric
+__lt__(other) BaseSingleMetric
+__len__() int
+sum() float
+mean() float
+count() int
+abs() BaseSingleMetric
+empty bool
+add(other, fill_value) BaseSingleMetric
+replace(replace_dict) BaseSingleMetric
+apply(func) BaseSingleMetric
}
class BaseOrderIndicator {
<<abstract>>
+__init__() void
+assign(col, metric) void
+transfer(func, new_col) Optional[BaseSingleMetric]
+get_metric_series(metric) pd.Series
+get_index_data(metric) SingleData
sum_all_indicators(order_indicator, indicators, metrics, fill_value) void
+to_series() Dict[Text, pd.Series]
}
class PandasSingleMetric {
+__init__(metric) void
+sum() float
+mean() float
+count() int
+abs() PandasSingleMetric
+empty bool
+add(other, fill_value) PandasSingleMetric
+replace(replace_dict) PandasSingleMetric
+apply(func) PandasSingleMetric
+reindex(index, fill_value) PandasSingleMetric
}
class PandasOrderIndicator {
+__init__() void
+assign(col, metric) void
+get_index_data(metric) SingleData
+get_metric_series(metric) pd.Series
+to_series() Dict[str, pd.Series]
sum_all_indicators(order_indicator, indicators, metrics, fill_value) void
}
class NumpyOrderIndicator {
+__init__() void
+assign(col, metric) void
+get_index_data(metric) SingleData
+get_metric_series(metric) pd.Series
+to_series() Dict[str, pd.Series]
sum_all_indicators(order_indicator, indicators, metrics, fill_value) void
}
BaseQuote <|-- PandasQuote
BaseQuote <|-- NumpyQuote
BaseSingleMetric <|-- PandasSingleMetric
BaseOrderIndicator <|-- PandasOrderIndicator
BaseOrderIndicator <|-- NumpyOrderIndicator
```

**图源**
- [high_performance_ds.py](file://qlib/backtest/high_performance_ds.py)

**节源**
- [high_performance_ds.py](file://qlib/backtest/high_performance_ds.py)