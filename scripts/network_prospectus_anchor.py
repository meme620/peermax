"""第二梯队 #9 路线 A（plan §8 B″11-7 锚 1）：招股书「同行业可比公司」自报锚试点。

口径真源 = `configs/parameter_registry.yaml` 的 `prospectus_peer_anchor_pilot`（P0 先登记后执行）
+ `reports/audits/s3_remaining_workplan_20261011.md` 第二梯队 #9 路线 A。

步骤（P2–P5 一体）：
  P2 解析定位：pymupdf 页码级扫描 4 个关键词（同行业可比公司 / 可比上市公司 / 同行业上市公司 /
     同行业可比上市公司），记录命中数、首个命中页码；`located = 命中数 > 0`。
  P3 抽名单：在命中窗口（hit−150 .. hit+1500，重叠合并）内按 5 条正则规则抽取公司名：
     R1 全称型（…股份有限公司/有限责任公司/有限公司）；R2 短名+代码型（短名（NNNNNN））；
     R3 代码+短名型（NNNNNN.SZ/SH 后随短名）；R4 列举型（包括/分别为/主要有… ×、×）；
     R5 表格行型（独立成行的 2–10 字短名，下一非空行以数字/百分号开头）。
     归一化：去空白/全角半角统一/去引号；去「股份有限公司/有限责任公司/有限公司」后缀得 base；
     排除发行人自身（全称与简称）、聚合词与通用词（黑名单）、纯数字/长度 <2 的 token。
  P4 映射：名称→代码，模式 = `exact_short`（项目 firm 名单简称精确）/ `exact_full`（全称字典精确）/
     `suffix_norm`（去公司后缀后精确）/ `adjacent_code`（窗口内自带 6 位 A 股代码）；失败 = 上述全不中。
     全称字典 = ① 年报文本头部全称（本地抽取，缓存 artifacts/tmp）+ ② 东财曾用名/全称缓存。
  P5 度量（主口径网络 = `pm_ma_ewcos_teacher_v1_{试点年+2}`，该 focal 上市后第一个其为节点的版本）：
     重合率 share = Σ_i #{j∈D_i : (i,j)∈E} / Σ_i #D_i；基线 d = E/(N(N−1)/2)（N = 入网公司含零对手，
     E = 无向边数，与 lift_pair_density_readout 同口径）；lift = share / d。

产出：`reports/data_gates/G0/s3_network_prospectus_pilot_<year>.json` + `.md`（+ 逐公司明细 CSV）。
停机（已登记）：解析成功率 <80% / 映射失败率 >20% / 覆盖率 <50% → 停（不得宽松匹配或人工补录）。

用法: python scripts/network_prospectus_anchor.py --year 2018
      python scripts/network_prospectus_anchor.py --year 2018 --codes 601138,300750"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
import pymupdf


import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

FIRM_MAP = os.path.join(ROOT, "data", "processed", "firm_id_map.csv")

A_STOCKS = os.path.join(ROOT, "data", "raw", "baostock", "stock_basic", "a_stocks.csv")

NAME_HIST = os.path.join(ROOT, "data", "processed", "name_history_refined.csv")

NAME_HIST_CAND = os.path.join(ROOT, "data", "processed", "name_history_candidates.csv")

EM_NAMES = os.path.join(ROOT, "data", "raw", "cninfo", "em_former_names.json")

ANNUAL_TXT = os.path.join(ROOT, "data", "raw", "cninfo", "annual_txt")

CACHE = os.path.join(ROOT, "artifacts", "tmp", "tier2_prosp", "full_name_map.json")

_WS = re.compile("[ \u3000\t\r\n]")   # 头部文本空白归一（含全角空格）

NET_DIR = os.path.join(ROOT, "data", "processed", "network_ma_teacher")

KWS = ("同行业可比公司", "可比上市公司", "同行业上市公司", "同行业可比上市公司")

KW_RE = re.compile("|".join(KWS))

WIN_PRE, WIN_POST = 100, 500   # 同行名单紧跟关键词标题（表格/列举在 500 字内）；过宽会引入无关章节实体


R1 = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9（）()]{2,22}?(?:股份有限公司|有限责任公司|有限公司)")

_R2_LABEL = r"(?:股[\s\u3000]*票[\s\u3000]*代[\s\u3000]*码|证[\s\u3000]*券[\s\u3000]*代[\s\u3000]*码" \
            r"|公[\s\u3000]*司[\s\u3000]*代[\s\u3000]*码)"

R2 = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})[（(][\s\u3000]*" + _R2_LABEL +
                r"?[\s\u3000]*[：:]?[\s\u3000]*(\d{6})[\s\u3000]*[）)]")

R3 = re.compile(r"(\d{6})\.(?:SZ|SH|sz|sh)[^\u4e00-\u9fa5]{0,25}?([\u4e00-\u9fa5A-Za-z0-9]{2,10})")

R4_PEER_TRIG = ("同行业可比上市公司", "同行业可比公司", "同行业上市公司", "同行业公司",
                "可比上市公司", "可比公司", "同业公司")

R4 = re.compile(r"(?:包括|分别为|分别如下|主要有|如下表所示|有：|为：|" + "|".join(R4_PEER_TRIG) + r")"
                r"[^\u4e00-\u9fa5]{0,10}"
                r"((?:[\u4e00-\u9fa5A-Za-z0-9]{2,10}[、,，]){1,}[\u4e00-\u9fa5A-Za-z0-9]{2,10})")

R4_SEP = re.compile(r"[、,，]|和|与|及")

TBL_HEAD = ("可比公司", "同行业可比公司", "同行业上市公司", "同行业可比上市公司",
            "可比上市公司", "同业公司", "同行业公司")

TBL_HDR_TOK = re.compile(r"^(?:[\d,\.\-]+%?|[\d]{4}\s*年|年|度|月|日|单位|万元|%|％)$")

R4_MEMBER_RE = re.compile(r"^[\u4e00-\u9fa5]{2,6}$")

R4_MEMBER_STOP = "的了一个较虽而但且很更最均将会能需应使被把让给向往于对从由比该各如则因所在之其也都又才就并非未无等及"

TBL_ROW = re.compile(r"^([\u4e00-\u9fa5]{2,10})[\s\u3000]+(\S.*)$")

NOISE_SUB = ("收入", "成本", "费用", "利润", "资产", "负债", "账款", "款项", "权益", "现金", "存货",
             "折旧", "摊销", "税额", "税收", "支出", "收益", "余额", "净额", "减值", "拨备",
             "比例", "占比", "金额", "合计", "平均", "增长", "变动", "其中", "分类", "正常类",
             "关注类", "次级类", "可疑类", "损失", "未披露", "无数据", "项目", "单位", "年度",
             "期间", "数据来源", "资料来源", "分公司", "子公司", "比重", "倍数", "天数", "周转",
             "毛利率", "净利率", "每股", "净资产", "总资产", "薪酬", "万元", "亿元", "个百分点",
             "员工人数", "注册地", "上市地", "法定代表人", "注册资本")

NOISE_CHAR = "占率额费税价值数项类比年月日是否"   # 仅对无后缀的表格行短名（R5）额外过滤

NOISE_WEAK = ("设备", "材料", "商品", "业务", "指标", "费用", "工资", "奖金", "补贴", "福利",
              "其他", "其它", "各类", "其中", "房屋", "建筑物", "库存", "用品", "人工", "劳务",
              "财务", "折旧", "摊销", "低值", "易耗", "原材料", "成品", "半成品", "工程款",
              "投资管理", "有限合伙", "基金", "中心", "研究院")

CONN_SUB = ("以及", "包括", "其中", "例如", "简称", "上市", "挂牌", "拟", "同业", "可比", "母子",
            "公司之", "系", "为", "是", "和", "与", "及", "的", "其", "等", "非", "新增", "收购",
            "客户", "供应商", "股东", "关联", "控股子", "参股", "股权", "投资管理合伙企业",
            "有限合伙", "子公司", "分公司")

NOISE_TAIL = ("之", "等", "及", "和", "与", "或", "为", "的", "指", "为：")

R5_NAME = re.compile(r"^[\u4e00-\u9fa5]{2,10}$")

R5_NUM = re.compile(r"^[（(]?-?[\d,\.]+%?[）)]?$|^[\d,\.]+$|^[-—]+$")

LABEL_CUT = ("转债代码", "债券代码", "港股代码", "转股代码", "转债简称", "债券简称", "股票代码",
             "证券代码", "公司代码", "股票简称", "证券简称", "公司简称", "中文简称", "英文简称",
             "公司全称", "债券全称", "英文名称", "上市日期", "上市地", "注册地址", "办公地址")

GEO_EXTRA = ("中国", "黑龙", "内蒙", "黑龙江", "内蒙古")   # H1-(a) 切分判据允许的「区域/中国」前缀补充

NOISE_TOKEN = {"占率", "比率", "占比", "比例", "数量", "金额", "费用", "税额", "税率", "价值",
               "项目", "类别", "类比", "年度", "月份", "年月", "是否", "合计", "小计", "平均",
               "均值", "天数", "倍数", "比重", "单位", "万元", "亿元", "百分点", "本期", "上期"}

R2_CTX_HEAD = ("同行业可比上市公司", "同行业可比公司", "同行业公司", "同业公司", "可比公司",
               "同行业", "同业", "可比", "包括", "以及", "主要有", "分别为", "其中", "例如",
               "和", "与", "及")

SUFFIX_END = re.compile(r"(?:股份有限公司|有限责任公司|有限公司)$")

FULL_NAME_RE = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)$")

NOISE_GENERIC = ("报告期", "季度", "一方面", "另一方面", "差旅费", "宣传费", "土地使用权", "运输工具",
                 "在产品", "委托加工", "行业标准", "应用领域", "消费电子", "研发人员", "销售数量",
                 "信息来源", "期内", "年度", "行业", "领域", "标准", "人员", "费用", "物资", "工具",
                 "设备", "材料", "商品", "业务", "服务", "其他", "其中", "价格", "数量", "金额")

GEN_SUB = NOISE_GENERIC

GEN_SUB_SET = set(GEN_SUB)

GEN_SUB_MAXLEN = 5                       # 「整名 ≈ 通用词」长度门槛（>5 字不再由本判据判读）

GEN_SUB_TAIL_EXEMPT = ("公司", "集团", "股份", "有限", "企业")   # 公司后缀豁免（收尾词）

RESIDUAL_CLS = "残差桶（未过前三类判据）"

RESIDUAL_EV = "未过碎片/名称漂移/抽取噪声三类判据（兜底残差；来源需独立判读，不得断言为「非 A 股实体」）"

_GEO = None

_SELF_IDX = {"full": {}, "short": {}, "cache": {}}

_CTX_SUFFIX = None

_CTX_LINK_TAIL = ("有",)     # 修 1(e)：语境尾连接词（名录守卫下剥离）

NAME_START_BAD = ("属于", "生产", "销售", "控股", "主要", "包括", "从事", "主营", "系", "为", "是",
                  "及", "和", "与", "的", "其", "公司", "业务", "产品", "行业", "领域", "经营", "研发",
                  "制造", "加工", "以及", "其中", "例如", "根据", "截至", "报告", "发行", "募集",
                  "使用", "实现", "具有", "存在", "通过", "采用", "提供", "形成", "达到", "增长")

NAME_BIZ_WORD = ("香精", "生长", "装备", "专用", "通用", "综合治", "集成电路", "半导体", "单晶硅",
                 "太阳能", "光伏")

ST_PREFIX = re.compile(r"^(?:\*ST|ST|S\*ST|SST|S)(?=[\u4e00-\u9fa5])")

REJECT_LOG = []   # 修 1/修 4：被 R3 形态约束与 R1/R2 交叉校验剔除的候选（供方向性检查）

REPAIR_LOG = []   # 修 5（registry 1.5.15）：R2 左边界修复记录（raw → fixed；证据，不进方向性检查）

BLACK = {
    "本公司", "公司", "发行人", "上述公司", "该公司", "标的公司", "股份公司", "上市公司",
    "同行业可比公司", "同行业上市公司", "同行业公司", "可比公司", "同行业", "可比",
    "行业平均", "行业平均值", "平均值", "均值", "可比公司均值", "平均", "合计", "小计", "总计",
    "项目", "单位", "金额", "占比", "比例", "年度", "数据来源", "资料来源", "注", "备注",
    "股票代码", "证券代码", "名称", "公司名称", "代码", "简称", "主营业务", "营业收入",
    "毛利", "毛利率", "期间费用", "流动比率", "速动比率", "资产负债率", "应收账款",
    "存货", "净利润", "总资产", "净资产", "研发费用", "销售费用", "管理费用", "财务费用",
    "子公司", "母公司", "控股股东", "实际控制人", "关联方", "客户", "供应商", "其他",
    "其中", "序号", "无", "是", "否", "本期", "上期", "报告期", "公司名称及简称",
    "员工薪酬", "万元", "亿元", "元", "个百分点", "薪酬", "增长率", "证券简称", "上市日期",
    "公司名称及简称", "招股说明书", "招股意向书", "年度报告",
}

SELF_RE = re.compile(r"(?:股份)?(?:有限责任公司|有限公司|股份有限公司)$")

_CID_CACHE = os.path.join(ROOT, "artifacts", "tmp", "tier2_prosp", "cid_recover")

_SIMSUN = "C:/Windows/Fonts/simsun.ttc"

_GID2UNI = None

_OCR_DPI = 200

_OCR_PSM = "3"

_OCR_LANG = "chi_sim"

_OCR_CHUNK = 80            # 每 chunk 页数（一个 wsl bash 循环）

_OCR_MAX_CHUNKS = 2        # 并行 chunk 数上限（**实测调优**：6 workers × 2 chunk 时 vCPU 不饱和；

_WSL_EXE, _WSL_USER = "wsl.exe", "27697"


_OCR_SH = os.path.join(ROOT, "scripts", "ocr_tesseract_chunk.sh")

FRAG_EXPLICIT = {"务服务", "信息来源", "公司华东地区", "公司行业竞争力不断增", "销售数量", "子技术",
                 "达通信", "日电梯", "坚瑞沃", "招股书", "生物质发电", "疆东方环宇燃气"}

DRIFT_EXPLICIT = {"亿晶光电", "亿阳信通", "正平股份", "坚瑞沃能", "东方航空"}

FRAG_WORD = ("招股书", "招股意向书", "说明书", "信息来源", "资料来源", "销售数量", "竞争力", "不断增",
             "华东地区", "生物质", "地区")

NAME_TAIL = ("股份", "科技", "电器", "软件", "生物", "激光", "化工", "医药", "实业", "网络", "物业",
             "光电", "材料", "机械", "数控", "智造", "国际", "电子", "传媒", "通信", "环保", "能源",
             "食品", "家居", "建筑", "装饰", "设备", "集团", "控股", "视像", "照明", "汽车", "电机",
             "电缆", "纸业", "纺织", "服装", "模具", "轴承", "阀门", "泵业", "仪表", "仪器")

CJK_CH = re.compile(r"[\u4e00-\u9fa5]")

PLACE_PREFIX = {"合肥", "宁波", "湖北", "福建", "广西", "湛江", "陕西", "邵武", "珠海", "南京", "商洛",
                "青岛", "无锡", "常州", "苏州", "杭州", "广东", "深圳", "上海", "北京", "天津", "重庆",
                "山东", "江苏", "浙江", "湖南", "四川", "河南", "河北", "安徽", "江西", "云南", "贵州",
                "山西", "辽宁", "吉林", "黑龙", "内蒙", "新疆", "西藏", "宁夏", "甘肃", "青海", "海南",
                "香港", "台湾", "厦门", "大连", "武汉", "成都", "西安", "济南", "烟台", "潍坊", "绍兴",
                "嘉兴", "台州", "温州", "金华", "泉州", "东莞", "佛山", "中山", "惠州", "汕头", "洛阳"}

NON_ENTITY_WORD = ("库区", "分部", "基地", "工厂", "分公司", "子公司", "办事处", "项目", "工程",
                   "公司之", "（", "(", "职工", "团队")

_MEAS = {}

CACHE_INFO = {}          # 名称注册表 info（本进程实建/缓存命中；修 5 证据字段用）
import os
import re

def r4_member_ok(t):
    """修 1(a) 守则①：成员 = 2–6 汉字、非动词/连接词/科目词起头、**全名不含虚词字符**。"""
    if not R4_MEMBER_RE.match(t):
        return False
    if any(t.startswith(x) for x in NAME_START_BAD):
        return False
    if any(c in R4_MEMBER_STOP for c in t):
        return False
    return True

def r4_strip_head(t, uni_keys):
    """修 1(a) 守则④（registry 1.5.17）：名录守卫的**首字修复**——t[1:] 在 A 股在册名录（uni_keys）
    且 t 本身不在册 → 剥首字（切分/拼接残片，如 `有海泰科→海泰科`、`中苏交科→苏交科`、`中广日股份→广日股份`）；
    真名（如 `有研新材` = 自身在册）不剥；剥后若落到在册名，则可被映射为同行（而非计入未映射）。"""
    s = t
    while len(s) > 2 and s not in uni_keys and s[1:] in uni_keys:
        s = s[1:]
    return s

def r4_peer_list_ok(toks, gap, uni_keys):
    """修 1(a) 守则②③（registry 1.5.17）：复合触发词直列**必须是名单**——
    ② 触发词与首位成员之间不得有并列分隔符（散文续写常见「…公司，公司主要…」）；
    ③ 名单须含 **≥1 个 A 股在册名**（`uni_keys` 名录守卫；纯散文/科目/账龄枚举无人名 → 弃）。
    方向 = 从严（只影响**新增**的复合触发路径；既有「包括/分别为/有：+ 同行语境」路径逐字未改）。"""
    if any(ch in gap for ch in "、,，"):     # 并列分隔符出现在触发词与首位成员之间 = 散文续写（；/： 允许）
        return False
    if not any(t in uni_keys for t in toks):
        return False
    return True

def is_generic_name(b):
    """修 3（registry 1.5.17）：抽取噪声判据 = 「整名 ≈ 通用词」+ 公司后缀豁免。

    ① **公司后缀豁免**：名以 公司/集团/股份/有限/企业 收尾，或含「股份」「集团」→ **不得**判为抽取噪声
       （原「子串命中」判据把真实公司名（上海弘陆物流设备 / 山东联创节能新材料 / 国网综合能源服务…）
       误判为噪声：复核实测 117 条中 73 条 ≥6 字或带 公司/集团/股份）；
    ② **整名 ≈ 通用词**：整名 ∈ 词表，或整名长度 ≤ `GEN_SUB_MAXLEN` 且含词表词
       （= 整名近似于通用词/科目词/句片段词，如 一季度 / 差旅费 / 在产品 / 应用领域 / 运输工具）；
    ③ 其余（多为句片段/公司名）**不由本判据判读** → 落入残差桶（需独立判读）。
    """
    if any(b.endswith(t) for t in GEN_SUB_TAIL_EXEMPT) or "股份" in b or "集团" in b:
        return False
    if b in GEN_SUB_SET:
        return True
    return len(b) <= GEN_SUB_MAXLEN and any(t in b for t in GEN_SUB)

def geo_prefix():
    """「区域/中国」前缀集（H1 切分判据用；延迟引用 §P5 的 PLACE_PREFIX，避免前向引用歧义）。"""
    global _GEO
    if _GEO is None:
        _GEO = set(PLACE_PREFIX) | set(GEO_EXTRA)
    return _GEO

def cut_label_tail(cap):
    """H1-(b)：标签边界切分——取捕获串中最早的标签片段位置切分（保留 ≥2 字，否则原样返回）。"""
    best = None
    for lb in LABEL_CUT:
        i = cap.find(lb)
        if i >= 2 and (best is None or i < best):
            best = i
    return cap[:best] if best is not None else cap

def trim_dup_name(seg, start, cap):
    """H1-(a)：年报头部「简称」捕获串规范化 = 标签边界切分 + 「简称+全称同头拼接」最短合法切分。

    切分 L 合法 ⟺ ① rest = ext[L:] 形态为 …股份有限公司/有限责任公司/有限公司；② rb = base_name(rest)
    与简称头部一致——rb 以简称前 2 字起头，**或** rb 以区域/「中国」前缀（2–3 字）起头且其后以简称
    前 2 字起头；③ rb 不以公司后缀残片（股份/有限/责任/公司）起头。取最小合法 L；无合法切分时返回
    标签切分后的捕获串（不改动）。**判据登记 = registry 1.5.15**。
    """
    cap = cut_label_tail(cap)
    if len(cap) < 2:
        return cap
    j, ext = start + len(cap), cap
    while j < len(seg) and len(ext) < 30 and not SUFFIX_END.search(ext):
        if not CJK_CH.match(seg[j]):
            break
        ext += seg[j]
        j += 1
    for L in range(2, min(10, len(ext)) + 1):
        short, rest = ext[:L], ext[L:]
        if not FULL_NAME_RE.match(rest):
            continue
        rb = base_name(rest)
        if rb.startswith(("股份", "有限", "责任", "公司")):
            continue
        head = short[:2]
        ks = [0] + [k for k in (2, 3) if rb[:k] in geo_prefix()]
        if any(len(rb) - k >= 2 and rb[k:].startswith(head) for k in ks):
            return short
    return cap

def ctx_suffix_frags():
    """修 1(e)（registry 1.5.17）：语境词的 **≥2 字后缀片段** 集（R2 捕获窗口 12 字截断时
    捕获串常以语境词的中段起头，如 `行业可比上市公司有东方雨虹` ← 「同行业可比上市公司有」）。"""
    global _CTX_SUFFIX
    if _CTX_SUFFIX is None:
        s = set()
        for p in R2_CTX_HEAD:
            for k in range(2, len(p) + 1):
                s.add(p[-k:])
        _CTX_SUFFIX = s
    return _CTX_SUFFIX

def repair_r2_left(raw, uni_keys=()):
    """M1-(a) + 修 1(e)：R2 左边界修复——语境/连接词前缀逐层剥离 + 语境后缀片段剥离 +
    尾随连接词（「有」）**名录守卫**剥离（修复轮二，registry 1.5.17）。

    守则（顺序）：
      ① 捕获串本身即在 A 股在册名录（uni_keys）→ **整体不剥离**（防误切真名）；
      ② 既有 R2_CTX_HEAD 逐层剥离（≥2 字剩余）；
      ③ 修 1(e)：以任一语境词的 ≥2 字**后缀片段**起头（剩余 ≥3 字）→ 剥离一次；
      ④ 剥离后回到 ② 再剥一层（露出连接词的情形）；
      ⑤ 修 1(e)：以「有」起头且剩余 ≥2 字、**整体不在册** → 剥离该「有」（名录守卫防误切「有研新材」）。
    """
    s = raw
    if raw in uni_keys:          # ① 名录守卫（uni_keys 为集合；在册名整体不剥离）
        return s
    changed = True
    while changed and len(s) > 4:
        changed = False
        for p in R2_CTX_HEAD:
            if s.startswith(p) and len(s) - len(p) >= 2:
                s, changed = s[len(p):], True
                break
    for L in range(min(12, len(s) - 3), 1, -1):     # 修 1(e)：语境后缀片段（保留 ≥3 字）
        if s[:L] in ctx_suffix_frags():
            s = s[L:]
            break
    changed = True
    while changed and len(s) > 4:
        changed = False
        for p in R2_CTX_HEAD:
            if s.startswith(p) and len(s) - len(p) >= 2:
                s, changed = s[len(p):], True
                break
    for t in _CTX_LINK_TAIL:
        if s.startswith(t) and len(s) - len(t) >= 2 and s not in uni_keys:
            s = s[len(t):]
    return s

def lev_leq1(a, b):
    """M2 容错：|Δlen| ≤1 且编辑距离 ≤1 的快速判据（O(n)）。"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(s) and j < len(l):
        if s[i] == l[j]:
            i, j = i + 1, j + 1
            continue
        if skipped:
            return False
        skipped, j = True, j + 1
    return True

def build_self_index(full, short_hist):
    """按 code 倒排名称注册表（H1/M2 修复轮：自名集合须含**代码锚定历史简称**；一次构建、全程复用）。"""
    fu, sh = {}, {}
    for k, v in full.items():
        for c in v:
            fu.setdefault(c, []).append(k)
    for k, v in short_hist.items():
        for c in v:
            sh.setdefault(c, []).append(k)
    _SELF_IDX.update(full=fu, short=sh, cache={})

def self_names_for(firm, code):
    """发行人自名集合 (exact, core)：universe 当前名 + 全称字典键 + **名称注册表历史简称**（均含基名）。

    core = 容错判据比较池 = exact 中长度 ≥2 者（**不设长度上限**——登记 M2 判据为「与**某自名**」，
    长度窗口会漏掉长全称（如「…股份有限公司」>14 字）上的同核/近全包含命中）。**登记 = registry 1.5.15**。
    """
    ent = _SELF_IDX["cache"].get(code)
    if ent is None:
        exact = {norm(firm[code]["name"]), base_name(firm[code]["name"])}
        for k in _SELF_IDX["full"].get(code, ()):
            exact.add(k)
            exact.add(base_name(k))
        for k in _SELF_IDX["short"].get(code, ()):
            exact.add(k)
            exact.add(base_name(k))
        exact = {x for x in exact if x}
        ent = (exact, {x for x in exact if len(x) >= 2})
        _SELF_IDX["cache"][code] = ent
    return ent

def is_self_name(b, raw, self_exact, self_core, uni_keys=()):
    """M2：发行人自指容错判据（登记 = registry 1.5.15）。

    判据（任一命中即判自指）：(i) 精确；(ii) 同核（与某自名共享 **≥3 字前缀，上限 4 字**——逐 k∈{4,3} 比对）；
    (iii) 编辑距离/OCR 近似（|Δlen| ≤1 且 lev ≤1，双方 ≥4 字）；(iv) 近全包含（b ⊆ 自名，残差 ≤4 字，len(b) ≥3）。
    守卫：b ∈ A 股 universe 名录（uni_keys）→ **不**做自指剔除（防误剔真实在册名）。
    """
    if not b:
        return False
    if b in self_exact or raw in self_exact:
        return True
    if b in uni_keys or len(b) < 3:
        return False
    for s in self_core:
        if b == s:
            return True
        if len(b) <= len(s) and b in s and len(s) - len(b) <= 4:
            return True
        for k in (4, 3):                       # 同核：共享 ≥3 字前缀（上限 4 字）
            if len(b) >= k and len(s) >= k and b[:k] == s[:k]:
                return True
        if len(b) >= 4 and len(s) >= 4 and lev_leq1(b, s):
            return True
    return False

def strip_st(name):
    """去 ST/*ST/S*/S 前缀标记（修 2）：universe 当前名与招股书当时名之间的 ST 前缀漂移。"""
    s = ST_PREFIX.sub("", name or "")
    return s if len(s) >= 2 else (name or "")

def r3_shape_ok(b):
    """R3 名称形态约束（修 1）：3–10 汉字 + 非动词/连接词起头 + 无业务描述词。"""
    if not re.match(r"^[\u4e00-\u9fa5]{3,10}$", b):
        return False
    if any(b.startswith(x) for x in NAME_START_BAD):
        return False
    if any(x in b for x in NAME_BIZ_WORD):
        return False
    return True

def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return p

def norm(s):
    s = re.sub(r"[ \u3000\t\r\n\u00a0]", "", s or "")
    s = s.replace("（", "(").replace("）", ")").replace("，", ",").replace("、", ",")
    s = re.sub(r'["\'“”‘’《》<>]', "", s)
    return s

def base_name(s):
    """去公司后缀；保留到「集团」级别（不删集团/科技/股份，避免过度归一化）。"""
    s = norm(s)
    s = re.sub(r"(?:集团)?(?:股份有限公司|有限责任公司|有限公司)$", "", s)
    return s
import os
import pymupdf
import re
import shutil
import subprocess
import time

def _gid2uni():
    """SimSun（宋体）GID → Unicode 反查表（修 3 备选通道用；子集字体无 ToUnicode 时，
    Identity-H 编码的 2 字节码 = GID，可经同版本宋体反查回真字符）。"""
    global _GID2UNI
    if _GID2UNI is None:
        from fontTools.ttLib import TTCollection
        f = TTCollection(_SIMSUN, lazy=True).fonts[0]
        rev = {}
        for u, g in sorted(f.getBestCmap().items()):
            gid = f.getGlyphID(g)
            if gid not in rev or u < rev[gid]:
                rev[gid] = u
        _GID2UNI = rev
    return _GID2UNI

def cid_recover_text(code, pdf_path):
    """修 3 备选提取通道：pdfminer.six 抽取 + `(cid:N)` → SimSun GID → Unicode 解码。

    适用对象 = 文本层 CID 编码损坏（无 ToUnicode CMap 的子集字体，pymupdf 读出 `chr(cid)` 乱码、
    「公司/发行人/万元」计数 = 0）。解码为**确定性字符还原**（非 OCR）。缓存
    artifacts/tmp/tier2_prosp/cid_recover/<code>.txt（键 = PDF 字节数），命中即复用。
    """
    os.makedirs(_CID_CACHE, exist_ok=True)
    cache = os.path.join(_CID_CACHE, f"{code}.txt")
    key = os.path.join(_CID_CACHE, f"{code}.key")
    size = str(os.path.getsize(pdf_path))
    if os.path.exists(cache) and os.path.exists(key) and open(key, encoding="utf-8").read().strip() == size:
        return open(cache, encoding="utf-8").read(), "pdfminer+cid_gid（缓存命中）"
    import pdfminer.high_level
    rev = _gid2uni()
    raw = pdfminer.high_level.extract_text(pdf_path)
    n_unmapped = [0]

    def sub(m):
        cid = int(m.group(1))
        if cid < 0x80:
            return chr(cid)
        u = rev.get(cid)
        if u is None:
            n_unmapped[0] += 1
            return "\ufffd"
        return chr(u)

    dec = re.sub(r"\(cid:(\d+)\)", sub, raw)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(dec)
    with open(key, "w", encoding="utf-8") as f:
        f.write(size)
    return dec, f"pdfminer+cid_gid（未映射 CID {n_unmapped[0]}）"

def _wsl_path(p):
    """Windows 绝对路径 → WSL /mnt/<drive>/... 路径。"""
    p = os.path.abspath(p)
    return f"/mnt/{p[0].lower()}" + p[2:].replace("\\", "/")

def _insufficient(text, n_pages):
    """文本层「缺失/不足」判据（registry 1.5.13）：`公司` 计数 < 20 或 字符数 < max(5000, 50×页数)。"""
    return text.count("公司") < 20 or len(text) < max(5000, 50 * n_pages)

def ocr_recover_text(code, pdf_path):
    """OCR 回退通道：pymupdf 渲染逐页 PNG → 分 chunk 并行 WSL tesseract → 按页序拼接。

    缓存 = <year_dir>/ocr_txt/{code}.txt（键 = PDF 字节数，命中即复用）；渲染 PNG 存
    artifacts/tmp/tier2_full/ocr_render/<code>/，用后即删（不随件留存）。
    返回 (text, detail_str)；detail_str 即 channel 记录（含页数/耗时/缺页）。
    """
    year_dir = os.path.dirname(os.path.dirname(os.path.abspath(pdf_path)))
    cache_dir = os.path.join(year_dir, "ocr_txt")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{code}.txt")
    keyf = os.path.join(cache_dir, f"{code}.key")
    size = str(os.path.getsize(pdf_path))
    if os.path.exists(cache) and os.path.exists(keyf) and open(keyf, encoding="utf-8").read().strip() == size:
        return open(cache, encoding="utf-8").read(), "ocr_tesseract_chi_sim（缓存命中）"
    tmpd = os.path.join(ROOT, "artifacts", "tmp", "tier2_full", "ocr_render", code)
    shutil.rmtree(tmpd, ignore_errors=True)
    os.makedirs(tmpd, exist_ok=True)
    t0 = time.time()
    try:
        doc = pymupdf.open(pdf_path)
        n_pages = len(doc)
        for i, pg in enumerate(doc):
            pg.get_pixmap(dpi=_OCR_DPI).save(os.path.join(tmpd, f"p{i:05d}.png"))
        doc.close()
        render_sec = time.time() - t0
        chunks = [(s, min(n_pages, s + _OCR_CHUNK)) for s in range(0, n_pages, _OCR_CHUNK)]
        script = _wsl_path(_OCR_SH)

        def run_chunk(se):
            """单 chunk = 一次 wsl.exe：bash <脚本> <dir> <s> <e> <lang> <psm>（避免内联引号被 Windows→WSL 改写）。"""
            s, e = se
            subprocess.run([_WSL_EXE, "-u", _WSL_USER, "bash", script, _wsl_path(tmpd),
                            str(s), str(e), _OCR_LANG, _OCR_PSM],
                           capture_output=True, timeout=7200)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(_OCR_MAX_CHUNKS, len(chunks)))) as ex:
            list(ex.map(run_chunk, chunks))
        # 缺页补跑一次（同脚本按缺失页连续区间重跑；仍缺则计入 missing）
        missing_idx = [i for i in range(n_pages)
                       if not os.path.exists(os.path.join(tmpd, f"p{i:05d}.txt"))]
        if missing_idx:
            runs = []
            for i in missing_idx:
                if runs and i == runs[-1][1]:
                    runs[-1][1] = i + 1
                else:
                    runs.append([i, i + 1])
            for s, e in runs:
                run_chunk((s, e))
        ocr_sec = time.time() - t0 - render_sec
        parts, missing = [], 0
        for i in range(n_pages):
            tp = os.path.join(tmpd, f"p{i:05d}.txt")
            if os.path.exists(tp):
                parts.append(open(tp, encoding="utf-8", errors="replace").read())
            else:
                missing += 1
                parts.append("")
        text = "\n".join(parts)
        with open(cache, "w", encoding="utf-8") as f:
            f.write(text)
        with open(keyf, "w", encoding="utf-8") as f:
            f.write(size)
        return text, (f"ocr_tesseract_chi_sim（{n_pages}p，render {render_sec:.0f}s + "
                      f"ocr {ocr_sec:.0f}s，缺页 {missing}）")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

def locate(txt_path, pdf_path, code=None):
    """返回 (located, n_hits, kw_counts, first_page, n_pages, spans, text, page_offsets, channel)。

    通道优先级（registry 1.5.13 全量轮预登记）：
      ① pymupdf 文本层（默认）；
      ② 文本层缺失/不足时 → pdfminer.six + SimSun GID 确定性还原（仅 CID 编码损坏件，沿用试点通道）；
      ③ 仍不足 → OCR 回退（WSL tesseract 5.3.4 chi_sim，--psm 3，200 DPI）。
    「不足」判据 = `公司` 计数 < 20 或 字符数 < max(5000, 50×页数)（在 ①② 之后判定）。
    channel = 最终通道记录；试点 2018 的 100 正常件（pymupdf）/3 CID 件（pdfminer+cid_gid）判定不变。
    """
    doc = pymupdf.open(pdf_path)
    pages = [pg.get_text() for pg in doc]
    n_pages = len(pages)
    doc.close()
    text = "".join(pages)
    channel = "pymupdf"
    text_replaced = False       # 修 4（registry 1.5.17）：文本是否已由 CID/OCR 通道替换（页码不再适用）
    counts = {k: text.count(k) for k in KWS}
    if _insufficient(text, n_pages):
        cid_done = False
        # ② CID 确定性还原：仅当存在疑似编码损坏的文本层（字符数 >2000；扫描件/无文本层直接走 ③）
        if code and len(text) > 2000:
            try:
                rel_text, cid_note = cid_recover_text(code, pdf_path)
                if not _insufficient(rel_text, n_pages):
                    text, channel, cid_done = rel_text, cid_note, True
                    text_replaced = True
                else:
                    channel = f"pdfminer+cid_gid(不足)"
            except Exception as e:
                channel = f"pdfminer+cid_gid(失败:{type(e).__name__})"
        # ③ OCR 回退（仅当仍不足）
        if not cid_done and _insufficient(text, n_pages):
            if code:
                try:
                    ocr_text, ocr_note = ocr_recover_text(code, pdf_path)
                    prefix = "" if channel == "pymupdf" else f"{channel}→"
                    if not _insufficient(ocr_text, n_pages):
                        text, channel = ocr_text, f"{prefix}{ocr_note}"
                        text_replaced = True
                    else:
                        channel = f"{prefix}ocr_insufficient（回退后仍不足；{ocr_note}）"
                except Exception as e:
                    channel = f"{channel}→text_layer_invalid:ocr_fail:{type(e).__name__}"
            else:
                channel = f"{channel}→text_layer_invalid:no_code"
        counts = {k: text.count(k) for k in KWS}
    offsets, acc = [], 0
    for p in pages:
        offsets.append(acc)
        acc += len(p)
    hits = [m.start() for m in KW_RE.finditer(text)]
    spans = []   # (start, end, anchor_hit_position)
    for h in sorted(hits):
        a, b = max(0, h - WIN_PRE), min(len(text), h + WIN_POST)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b, h])
    first_page = None
    if hits:
        if text_replaced:
            # 修 4（registry 1.5.17）：CID 还原 / OCR 通道无 pymupdf 页码映射 → 标 not_applicable_channel
            #（不得以 pymupdf 页码冒充；pymupdf 页码与替换后文本的字符偏移不对应）
            first_page = "not_applicable_channel"
        else:
            h0 = hits[0]
            first_page = max(1, sum(1 for o in offsets if o <= h0))
    return bool(hits), len(hits), counts, first_page, n_pages, spans, text, offsets, channel

def _raw_ok(raw, rule):
    """token 形态过滤：公司名不得含连接词/数字/括号残片、不得以非中文字符开头。"""
    b = raw.strip()
    if re.match(r"^(ST|\*ST|S\*ST|SST)", b):
        b = re.sub(r"^(ST|\*ST|S\*ST|SST)", "", b)
    if not b or not re.match(r"^[一-龥A-Za-z]", b):
        return None
    if re.search(r"[0-9（）()\[\]【】]", b):
        return None
    if any(c in b for c in CONN_SUB):
        return None
    if len(b) < 2 or len(b) > 20:
        return None
    if rule in ("R1_full", "R5_table_row", "R4_enum") and len(b) < 3:
        return None
    return b

def extract_candidates(text, spans, self_exact, self_core, uni_keys=()):
    """返回 [dict(raw, base, rule, adjacent_code)]（未去重）。

    精度设计（解析层，抽取规则；映射规则不变）：
      R1 全称型：形态由「股份有限公司/有限责任公司/有限公司」后缀保证，命中窗口内直接采纳；
      R2/R3 代码锚定型：形态与语境同时由显式 A 股代码保证；
      R4 列举型：仅当**同句前置 ≤40 字**内出现 同行业/可比/同业 语境词时采纳（否则「包括…」会命中科目枚举）；
        修 1(a)(b)（registry 1.5.17）：触发词补「可比公司」类复合词（触发词自证）+ 并列切分补 和/与/及；
      R4 表头列型（R4_tbl_head，修 1(f)，registry 1.5.17）：表头行首 token = 可比公司类词 + 行内其余 token
        均期间/数字类 → 取其后连续 ≥2 行（行首短名 + 行内含数字）的行首名；
      R5 表格行型（R5'）：先切成「表格块」= ≥2 个连续「短名行 + 数字/百分号行」，且**块前 120 字**含
         同行业/可比/同业/证券代码/股票代码 语境词（排除「纳税主体名称/所得税税率」这类同形非同行表）。
    """
    PEER_CTX = ("同行业", "可比", "同业", "证券代码", "股票代码")
    out = []
    for a, b, hit in spans:
        seg = text[a:b]
        hit_off = hit - a
        for m in R2.finditer(seg):
            out.append(dict(raw=m.group(1), base=base_name(m.group(1)), rule="R2_name_code",
                            adjacent_code=m.group(2)))
        for m in R3.finditer(seg):
            out.append(dict(raw=m.group(2), base=base_name(m.group(2)), rule="R3_code_name",
                            adjacent_code=m.group(1)))
        for m in R1.finditer(seg):
            if not (-WIN_PRE <= m.start() - hit_off <= 250):   # 全称型只取命中近邻（防远距表夹带）
                continue
            out.append(dict(raw=m.group(0), base=base_name(m.group(0)), rule="R1_full", adjacent_code=None))
        for m in R4.finditer(seg):
            pre = seg[max(0, m.start() - 20):m.start()]   # 列举型必须紧跟同行语境词（否则命中科目枚举）
            ctx_ok = any(k in pre for k in PEER_CTX)
            peer_trig = m.group(0).startswith(R4_PEER_TRIG)
            # 修 1(a)（registry 1.5.17）：触发词本身为「可比公司」类复合词时按触发词自证；**凡复合触发词路径一律加
            # 名单形态守则**（≥2 成员 ∧ 每成员形如公司简称 ∧ 名单含 ≥1 个 A 股在册名）——防散文续写
            #（「同行业可比公司不存在明显差异，主要由于…」「可比公司上市后发展较快，盈利能力、…」）入名单。
            if not ctx_ok and not peer_trig:
                continue
            # 修 1(b)：并列切分补 和/与/及——**仅复合触发路径**（既有「包括/分别为/有：+ 同行语境」路径逐字未改，
            # 防「产品包括 A 和 B（产品名词组）」被拆出伪名，如「层压焊接装框和检测设备」）。
            raw_toks = ([t for t in R4_SEP.split(m.group(1)) if t] if peer_trig
                        else [t for t in re.split(r"[、,，]", m.group(1)) if t])
            toks = raw_toks
            if peer_trig:
                gap = m.group(0)[:len(m.group(0)) - len(m.group(1))]
                good = [t for t in toks if r4_member_ok(t)]
                # ① 任一成员 >6 字（散文/句片段形态，非换行截断残片）→ 整名丢弃；
                # ② 名单须 ≥2 个形态合规成员 ∧ 触发词后无并列分隔符 ∧ 含 ≥1 个 A 股在册名（名录守卫）；
                # ③ 只把形态合规成员写入候选（≤2 字 OCR 断行残片与含虚词片段不入名单）；
                # ④ 名录守卫的**首字修复**：t[1:] 在 A 股在册名录且 t 本身不在册 → 剥首字（切分残片，如
                #    有海泰科→海泰科、中苏交科→苏交科；真名如 有研新材 因 t[1:] 不在册而不剥）。
                toks = [r4_strip_head(t, uni_keys) for t in good]
                if any(len(t) > 6 for t in toks) or not r4_peer_list_ok(toks, gap, uni_keys):
                    continue
            for tok in toks:
                out.append(dict(raw=tok, base=base_name(tok), rule="R4_enum", adjacent_code=None))
        # ---- R5' 表格块 ----
        lines, pos = [], []
        start = 0
        for ln in seg.split("\n"):
            lines.append(ln.strip())
            pos.append(start)
            start += len(ln) + 1
        rows = [i for i, ln in enumerate(lines)
                if R5_NAME.match(ln) and i + 1 < len(lines)
                and (lambda nx: bool(nx) and bool(R5_NUM.match(nx)))(
                    re.sub(r"[ \u3000]", "", next((x for x in lines[i + 1:i + 4] if x), "")))]
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1] - rows[j] <= 4:
                j += 1
            block = rows[i:j + 1]
            if len(block) >= 2 and 0 <= pos[block[0]] - hit_off <= 350:
                ctx = seg[max(0, pos[block[0]] - 60):pos[block[0]]]   # 块前 60 字内须有同行语境词
                # 兄弟前缀过滤：≥3 行共享 ≥4 字前缀 = 子公司/客户分部表（非同行名单）
                toks = [lines[r] for r in block]
                pref = ""
                if len(toks) >= 3:     # 兄弟前缀：≥3 行共享 4-8 字前缀 → 子公司/客户分部表
                    for L in range(min(len(toks[0]), 8), 3, -1):
                        cand = toks[0][:L]
                        if sum(1 for t in toks if t.startswith(cand)) >= 3:
                            pref = cand
                            break
                if any(k in ctx for k in PEER_CTX):
                    for r in block:
                        tk = lines[r]
                        if pref and len(pref) >= 4 and tk.startswith(pref):
                            continue
                        out.append(dict(raw=tk, base=base_name(tk),
                                        rule="R5_table_row", adjacent_code=None))
            i = j + 1
        # ---- 修 1(f)（registry 1.5.17）：R4_tbl_head「可比公司表头列」型 ----
        # 表头行（行首 token ∈ 可比公司类词，其余 token 均期间/数字类）后**连续** ≥2 行
        # （行首 2–10 汉字短名 + 行内含数字/百分号）→ 取行首名；走与 R4_enum/R5_table_row 相同的弱证据过滤。
        for hi, hln in enumerate(lines):
            htoks = hln.split()
            if not htoks:
                continue
            if htoks[0].strip("：:，,、") not in TBL_HEAD:
                continue
            if any(not TBL_HDR_TOK.match(t) for t in htoks[1:]):
                continue
            names = []
            for hj in range(hi + 1, min(hi + 26, len(lines))):
                rm = TBL_ROW.match(lines[hj])
                if not rm or not re.search(r"[\d%％]", rm.group(2)):
                    break
                names.append(rm.group(1))
            if len(names) >= 2:
                for nm in names:
                    out.append(dict(raw=nm, base=base_name(nm), rule="R4_tbl_head", adjacent_code=None))
    # 归一化过滤
    clean = []
    # 修 1：R3 与 R1/R2 交叉校验（R3 名不得是本文档 R1/R2 候选名的真片段）
    # 修 5（registry 1.5.15）：R2 左边界修复（语境/连接词前缀逐层剥离）**先行**，再算 strong 集合
    for c in out:
        if c["rule"] == "R2_name_code":
            r = repair_r2_left(c["raw"], uni_keys)      # 修 1(e)：加名录守卫（registry 1.5.17）
            if r != c["raw"] and len(r) >= 2:
                REPAIR_LOG.append(dict(raw=c["raw"], fixed=r))
                c["raw"], c["base"] = r, base_name(r)
    strong = {c["base"] for c in out if c["rule"] in ("R1_full", "R2_name_code")}
    for c in out:
        if _raw_ok(c["raw"], c["rule"]) is None:      # 形态过滤：连接词/数字/括号残片/句片段
            continue
        b = c["base"]
        if c["rule"] == "R3_code_name":
            if not r3_shape_ok(b):                    # 修 1：R3 名称形态约束（长度/起头词/业务描述词）
                REJECT_LOG.append(dict(base=b, raw=c["raw"], code=c["adjacent_code"],
                                       why="形态约束（长度/起头词/业务描述词）"))
                continue
            if any(s != b and b in s for s in strong):  # 修 1：与 R1/R2 交叉校验（真片段 → 弃）
                REJECT_LOG.append(dict(base=b, raw=c["raw"], code=c["adjacent_code"],
                                       why="R1/R2 交叉校验（真片段）"))
                continue
        if c["rule"] == "R2_name_code" and any(b.startswith(x) for x in NAME_START_BAD):
            REJECT_LOG.append(dict(base=b, raw=c["raw"], code=c["adjacent_code"], why="R2 起头词约束"))
            continue                                  # 修 1：R2 名同受起头词约束（如「控股公司奥瑞德」）
        if len(b) < 2 or not re.search(r"[\u4e00-\u9fa5]", b):
            continue
        if b in BLACK or c["raw"] in BLACK:
            continue
        if is_self_name(b, c["raw"], self_exact, self_core, uni_keys):
            REJECT_LOG.append(dict(base=b, raw=c["raw"], code=None,
                                   why="发行人自指（容错：精确/同核/编辑距离/近全包含）"))
            continue                                  # 修 5（M2）：自名过滤加容错，逐条留证据
        if any(s in b for s in NOISE_SUB):
            continue
        if c["rule"] in ("R4_enum", "R5_table_row", "R4_tbl_head"):
            if any(s in b for s in NOISE_WEAK):       # 弱证据规则（无代码锚定）加严
                continue
            if c["rule"] in ("R5_table_row", "R4_tbl_head"):
                # 修 5（M1②）：NOISE_CHAR 单字判据 → **整名判据**（不得因单字命中丢整名，如 红日药业）
                if b in NOISE_TOKEN or all(ch in NOISE_CHAR for ch in b):
                    continue
                if not R5_NAME.match(b):
                    continue
        if b.endswith(NOISE_TAIL) or b.startswith(NOISE_TAIL):
            continue
        if c["rule"] == "R1_full" and len(b) < 3:
            continue
        clean.append(c)
    return clean

def prefer_code(codes, universe, at_year=None):
    """多代码命中时的择一规则：优先 试点年当时已上市且未退市 的公司，其次最小代码。"""
    cs = sorted(codes)
    if at_year:
        act = [c for c in cs if universe[c]["ipo_date"] and int(universe[c]["ipo_date"][:4]) <= at_year
               and (not universe[c].get("out_date") or int(universe[c]["out_date"][:4]) >= at_year)]
        if act:
            return act[0]
    return cs[0]

def map_name(cand, short, full, universe, at_year=None):
    n = cand["base"]
    raw = cand["raw"]
    code = cand.get("adjacent_code")
    if code and code in universe:
        return code, "adjacent_code"
    for key in (norm(raw), n, strip_st(norm(raw)), strip_st(n)):
        if key in short:
            return prefer_code(short[key], universe, at_year), "exact_short"
        if key in full:
            return prefer_code(full[key], universe, at_year), "exact_full"
    # 后缀归一化：抽出的全称去后缀后与全称字典/简称字典的「去后缀」形式比对
    if n in full:
        return prefer_code(full[n], universe, at_year), "exact_full"
    alt = re.sub(r"集团$", "", n)
    for key in (alt, n):
        for cand_key in (alt, n):
            if cand_key in full:
                return prefer_code(full[cand_key], universe, at_year), "suffix_norm"
        if key in short:
            return prefer_code(short[key], universe, at_year), "suffix_norm"
    return None, "unmapped"

def _is_truncated(text, raw):
    """该 token 在正文中是否**总是**前接汉字（= 从更长中文串中间切出来的残片）。"""
    if not text or not raw:
        return False
    hits = 0
    for m in re.finditer(re.escape(raw), text):
        hits += 1
        if m.start() == 0 or not CJK_CH.match(text[m.start() - 1]):
            return False
        if hits >= 5:
            break
    return hits > 0

def classify_unmapped(b, raw, text, uni_keys, self_names=()):
    """返回 (类别, 证据)。类别 ∈ {碎片, 名称漂移, 抽取噪声, 残差桶（未过前三类判据）}。

    判据（**描述性口径**，逐名留证据；主口径失败率不依赖本分类）：
      碎片 = 复核清单 / 起头「公司·发行人」/ 含体裁·科目词 / 正文中总以前接汉字出现（截断残片）/
             为发行人自身全称的截断自指；
      名称漂移 = 与 A 股 universe 名**同头 ≥3 字**，或（同头 2 字 ∧ 公司式收尾词 ∧ 长度 ≥4）；
      抽取噪声 = 整名 ≈ 通用词/科目/句片段词（`GEN_SUB` **整名判据** + **公司后缀豁免**，修 3，registry 1.5.17；
             原「子串命中」判据把真实公司名（上海弘陆物流设备 / 山东联创节能新材料…）误判为噪声）；
      残差桶（未过前三类判据）= 其余——**兜底 return**，其来源需独立判读，**不得**以本桶断言「非 A 股实体」
             （修 2 改名，registry 1.5.17；复核抽样 n=160：真非上市/境外 ≈62.5%、抽取伪影 ≈30.0%、
             A 股名漂移 ≈7.5%）。
    """
    if b in FRAG_EXPLICIT:
        return "碎片", "复核清单（形态实测：截断/句片段/体裁词）"
    if b in DRIFT_EXPLICIT:
        return "名称漂移", "复核清单（A 股改名/ST 前缀漂移）"
    if any(b and b in s for s in self_names):
        return "碎片", "发行人自身名称的自指（抽取伪影，非外部同行）"
    if b.startswith(("公司", "本公司", "发行人", "上述", "该", "其")):
        return "碎片", f"起头为「{b[:2]}」"
    if any(w in b for w in FRAG_WORD):
        return "碎片", "含体裁/科目词"
    if _is_truncated(text, raw):
        return "碎片", "正文中总以「前接汉字」出现（截断残片）"
    for L in (3, 2):
        if len(b) > L and any(k.startswith(b[:L]) and k != b for k in uni_keys):
            if L >= 3 and not any(w in b for w in NON_ENTITY_WORD):
                return "名称漂移", f"A 股 universe 有同头 3 字名（{b[:L]}…）"
            if L == 2 and len(b) >= 4 and b[-2:] in NAME_TAIL and b[:2] not in PLACE_PREFIX:
                return "名称漂移", f"A 股 universe 有同头 2 字名（{b[:L]}…）+ 公司式收尾"
    if is_generic_name(b):     # 修 3（registry 1.5.17）：整名 ≈ 通用词 + 公司后缀豁免
        return "抽取噪声", "整名 ≈ 通用词/科目/句片段词（GEN_SUB 整名判据 + 公司后缀豁免；抽取层伪影，非外部实体）"
    # 修 2（registry 1.5.17）：第 4 类改名——原「非 A 股实体」是兜底 return 的**断言**而非判据证据；
    # 中性命名「残差桶（未过前三类判据）」，来源需独立判读（复核抽样 n=160：真非上市/境外 ≈62.5%、
    # 抽取伪影 ≈30.0%、A 股名漂移 ≈7.5%）。
    return RESIDUAL_CLS, RESIDUAL_EV
from datetime import datetime, timezone
import csv
import json
import os
import re
import time

def load_firm_universe():
    firm, short, ipo = {}, {}, {}
    with open(FIRM_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            firm[r["code6"]] = dict(firm_id=r["firm_id"], name=r["name"], ipo_date=r["ipo_date"],
                                    board=r["board"])
            short.setdefault(norm(r["name"]), set()).add(r["code6"])
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r["code"].split(".")[-1]
            if len(c) == 6 and c.isdigit() and c in firm:
                short.setdefault(norm(r["code_name"]), set()).add(c)
    for path in (NAME_HIST, NAME_HIST_CAND):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c = re.sub(r"\D", "", r.get("code", ""))[:6]
                nm = norm(r.get("name", ""))
                if c in firm and nm and len(nm) >= 2:
                    short.setdefault(nm, set()).add(c)
    # 修 2（2026-10-13 修正轮）：universe 名称去 ST/*ST/S*/S 前缀标记建键——试点年（2018）招股书用
    # 当时简称，universe 为当前简称；ST 前缀（如 *ST亿晶 / ST信通）会挡住历史简称的精确匹配。
    for k, v in list(short.items()):
        sk = strip_st(k)
        if sk != k:
            short.setdefault(sk, set()).update(v)
    return firm, short

def load_universe_name_keys():
    """A 股 universe 名称键（firm_id_map 当前简称 + a_stocks 简称 + 各自去 ST 前缀变体）；
    仅用于 unmapped 三分类的「名称漂移」判读（不含年报派生的第三方/子公司名）。"""
    keys = set()
    with open(FIRM_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = norm(r["name"])
            keys.add(n)
            keys.add(strip_st(n))
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = norm(r["code_name"])
            keys.add(n)
            keys.add(strip_st(n))
    return {k for k in keys if len(k) >= 2}

def build_full_name_map(firm, use_cache=True):
    """名称注册表：① 年报头部「股票简称/证券简称 + 股票代码」配对（历史简称，代码锚定）；
    ② 年报头部公司全称（最新与最早各一年）；③ 年报「证券简称由 X 变更为 Y」更名披露；
    ④ 年报释义段「X 指 Y（…公司）」缩写定义。缓存 artifacts/tmp/tier2_prosp/full_name_map.json（v3）。

    历史简称是**代码锚定**抽取（简称与 6 位代码相邻，或同属该代码自己的年报），非模糊匹配；用于弥合
    「项目 firm 名单为当前简称、招股书用当时简称」的名称漂移（如 良信电器→良信股份、东方航空→中国东航、
    坚瑞沃能→保力新）。**修 2（2026-10-13 修正轮）**：① 头部窗口 6000→24000 字（「公司股票简况」块常在
    6000 字之后）；② 标签补 `公司代码`/`A股证券简称` 且允许冒号；③ 头部「简称+全称同头拼接」去重；
    ④ 新增更名披露与释义缩写两条代码锚定来源；⑤ 注册名统一去 ST/*ST/S*/S 前缀标记建键。
    **修 5（2026-10-14 修复轮，registry 1.5.15）**：① 「简称+全称同头拼接」切分判据重写
    （`trim_dup_name`：最短合法切分 + rest 去后缀基名与简称头部一致性 + 公司后缀残片起头否决）；
    ② 新增**标签边界切分**（`cut_label_tail`：转债代码/债券代码/港股代码 等标签片段按最早边界切分）；
    ③ 缓存 `cache_version` 3→**4**（强制重算）。历史简称来源与代码锚定机制不变，**无人工补录**。
    """
    if use_cache and os.path.exists(CACHE):
        try:
            with open(CACHE, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("info", {}).get("cache_version") == 4 and d["info"].get("n_codes", 0) >= 4000:
                return ({k: set(v) for k, v in d["full"].items()},
                        {k: set(v) for k, v in d.get("short_hist", {}).items()}, d["info"])
        except Exception:
            pass
    full, short_hist, n_codes, n_files, n_short = {}, {}, 0, 0, 0
    n_change, n_defn = 0, 0
    codes = sorted(firm)
    t0 = time.time()
    P_SHORT_FWD = re.compile(r"(?:股票简称|证券简称|公司简称|中文简称|A股证券简称)[：:]?"
                             r"([一-龥A-Za-z0-9*]{2,10}?)[（(]?(?:股票代码|证券代码|公司代码)[：:]?([0-9]{6})")
    P_SHORT_REV = re.compile(r"(?:股票代码|证券代码|公司代码)[：:]?([0-9]{6})[^一-龥]{0,4}"
                             r"(?:股票简称|证券简称|公司简称|中文简称)[：:]?([一-龥A-Za-z0-9*]{2,10})")
    P_FULL = re.compile(r"[一-龥]{2,20}(?:股份有限公司|有限责任公司|有限公司)")
    # ③ 更名披露：「证券简称由“东方航空”变更为“中国东航”」（该年报所属代码 = 新旧名同属）
    P_NAME_CHANGE = re.compile(r"(?:A股)?(?:证券简称|股票简称)由[“\"']?([一-龥A-Za-z0-9*]{2,10})[”\"']?"
                               r"(?:变更为|更改为)[“\"']?([一-龥A-Za-z0-9*]{2,10})")
    # ④ 释义缩写：「东方航空指中国东方航空股份有限公司」（左名须为右全称的后缀/本身，排除子公司前缀式）
    P_DEFN = re.compile(r"([一-龥A-Za-z0-9]{2,10})指([一-龥A-Za-z0-9]{2,20}(?:股份有限公司|有限责任公司|有限公司))")

    def reg_short(nm, cd):
        """注册短名（含去 ST 前缀变体）。"""
        nonlocal n_short
        if not (cd in firm and 2 <= len(nm) <= 10):
            return False
        short_hist.setdefault(nm, set()).add(cd)
        n_short += 1
        sk = strip_st(nm)
        if sk != nm and 2 <= len(sk) <= 10:
            short_hist.setdefault(sk, set()).add(cd)
        return True

    def trim_dup(seg, start, cap):
        """修 5（registry 1.5.15）：转调模块级 `trim_dup_name`（切分判据 + 标签边界切分）。"""
        return trim_dup_name(seg, start, cap)

    for k, c in enumerate(codes):
        d = os.path.join(ANNUAL_TXT, c)
        if not os.path.isdir(d):
            continue
        years = sorted([x for x in os.listdir(d) if x.endswith(".txt")])
        got = 0
        for fn in years:
            n_files += 1
            try:
                head = re.sub(_WS, "", open(os.path.join(d, fn), encoding="utf-8",
                                            errors="replace").read(24000))
            except Exception:
                continue
            for m in P_SHORT_FWD.finditer(head):
                nm, cd = norm(cut_label_tail(m.group(1))), m.group(2)   # 修 5：标签边界切分
                if reg_short(nm, cd):
                    got += 1
            for m in P_SHORT_REV.finditer(head):
                cd, nm = m.group(1), norm(m.group(2))
                nm = trim_dup(head, m.start(2), nm)                      # 修 5：切分判据重写 + 标签边界切分
                if reg_short(nm, cd):
                    got += 1
            # 本公司全称（头部 1200 字内）——更名披露/释义缩写两条来源须指向**本公司**（防同业/客户/子公司误挂）
            own = {base_name(x.group(0)) for x in P_FULL.finditer(head[:1200])}
            for m in P_NAME_CHANGE.finditer(head):
                old, new = norm(m.group(1)), norm(m.group(2))
                if not own or not any((old in o or o in old or new in o or o in new) for o in own):
                    continue
                for nm in (old, new):     # 旧名与新名都登记（同代码）
                    if reg_short(nm, c):
                        n_change += 1
            for m in P_DEFN.finditer(head):
                left, right = norm(m.group(1)), norm(m.group(2))
                rb = base_name(right)
                if not (rb == left or rb.endswith(left)) or len(left) < 2:
                    continue
                if not own or not any((left in o or o in left) for o in own):
                    continue              # 释义缩写须指向本公司（否则是定义同业/客户）
                if any(x in left for x in CONN_SUB) or any(x in left for x in NOISE_SUB) \
                        or left in BLACK or left.startswith(NAME_START_BAD):
                    continue
                if reg_short(left, c):
                    n_defn += 1
            if fn in (years[-1], years[0]):
                for m in P_FULL.finditer(head):
                    nm = norm(m.group(0))
                    if len(nm) >= 6:
                        full.setdefault(nm, set()).add(c)
                        got += 1
        if got:
            n_codes += 1
        if (k + 1) % 1000 == 0:
            print(f"[P-2] 名称注册表构建 {k + 1}/{len(codes)}（覆盖 {n_codes} 只 / 简称条目 {n_short}，"
                  f"{time.time() - t0:.0f}s）", flush=True)
    if os.path.exists(EM_NAMES):
        try:
            em = json.load(open(EM_NAMES, encoding="utf-8"))
            for c, names in em.items():
                c6 = str(c).zfill(6)
                if c6 not in firm:
                    continue
                for nm in names:
                    nn = norm(nm)
                    if re.search(r"(?:股份有限公司|有限责任公司|有限公司)$", nn):
                        full.setdefault(nn, set()).add(c6)
                    elif 2 <= len(nn) <= 10:
                        reg_short(nn, c6)
        except Exception:
            pass
    # 注册名统一去 ST 前缀建键（修 2）：头部简称可能带 ST（如 ST信通），招股书用无前缀历史名
    for nm in list(short_hist):
        sk = strip_st(nm)
        if sk != nm:
            short_hist.setdefault(sk, set()).update(short_hist[nm])
    info = {"cache_version": 4, "n_codes": n_codes, "n_files": n_files, "n_full_names": len(full),
            "n_short_hist": len(short_hist), "n_short_pairs": n_short,
            "n_name_change_hits": n_change, "n_defn_hits": n_defn, "head_chars": 24000,
            "fix_round": "registry 1.5.15：trim_dup_name（最短合法切分 + 区域前缀一致性 + 后缀残片否决）"
                         "+ cut_label_tail（标签边界切分）",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 1),
            "source": "annual_txt 头部（股票简称/公司简称+股票代码 配对（标签/顺序/冒号变体、去 ST 前缀）；"
                      "全称取最新/最早各一年）+ 更名披露 + 释义缩写 + em_former_names.json"}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump({"info": info, "full": {k: sorted(v) for k, v in full.items()},
                   "short_hist": {k: sorted(v) for k, v in short_hist.items()}}, f, ensure_ascii=False)
    return full, short_hist, info
import os

def write_md(r, path):
    """人读版报告（P6）：三项率 / lift 表 / 抽样口径 / 失败案例 / 全量可行性判断。"""
    sc, ra, an = r["scope"], r["rates"], r["anchor_readout"]
    ov, nb = an["per_firm_overlap_stats"], an["network"]
    L = []
    A = L.append
    A(f"# 第二梯队 #9 路线 A 试点：招股书「同行业可比公司」自报锚（锚 1 / plan §8 B″11-7）")
    A("")
    A(f"- 运行时间（UTC）：{r['run_at']}；口径参数：`{r['parameter']}`")
    A(f"- 试点年 **{sc['pilot_year']}**（该年 IPO {sc['cohort_n']} 家 ≥100）；抽样口径 = **{sc['sample_scope']}**")
    A(f"- 网络版本：`{sc['network_version']}`（{sc['network_version_rule']}；available_at {sc['available_at']}）")
    A(f"- 招股书目录：`{sc['download']['dir']}`（pdf/ + txt/ + `download_log.csv` + `download_summary.json`）")
    A("")
    A("## 1. 三项实测率与停机判定")
    A("")
    A("| 指标 | 实测 | 分子/分母 | 停机线 | 判定 |")
    A("|---|---|---|---|---|")
    A(f"| 解析成功率 | **{ra and r['rates']['parse_success_rate']:.4f}** | {r['rates']['n_located']}/"
      f"{r['rates']['n_downloaded']}（下载成功份数） | ≥0.80 | "
      f"{'PASS' if r['rates']['parse_success_rate'] >= 0.80 else 'FAIL'} |")
    A(f"| 映射失败率 | **{r['rates']['mapping_failure_rate']:.4f}** | {r['rates']['n_unmapped_names']}/"
      f"{r['rates']['n_unique_names']}（去重唯一名称） | ≤0.20 | "
      f"{'PASS' if r['rates']['mapping_failure_rate'] <= 0.20 else 'FAIL'} |")
    A(f"| 覆盖率 | **{r['rates']['coverage_rate']:.4f}** | {r['rates']['n_covered']}/"
      f"{r['rates']['cohort_n']}（试点年 IPO ∩ 项目 universe） | ≥0.50 | "
      f"{'PASS' if r['rates']['coverage_rate'] >= 0.50 else 'FAIL'} |")
    A("")
    A(f"**停机判定 = {'触发（' + '；'.join(r['rates']['stop_reasons']) + '）' if r['rates']['stop_triggered'] else '未触发（三项全过）'}**")
    A("")
    A("覆盖率拆解：" + "；".join(f"{k} = {v}" for k, v in r["rates"]["coverage_breakdown"].items()))
    A(f"；网络节点内覆盖率 = {r['rates']['coverage_rate_in_network']:.4f}"
      f"（{r['rates']['n_network_nodes']} 家在网公司中已映射）。")
    A(f"每份已定位招股书平均抽出自报同行（已映射）= {r['rates']['peers_per_firm_mean']:.2f} 家；"
      f"映射模式计数 = {r['rates']['mapping_mode_totals']}。")
    A("")
    fd = r["rates"].get("failure_decomposition")
    if fd:
        A("### 1.1 映射失败率口径分列（碎片 / 名称漂移 / 残差桶（未过前三类判据））")
        A("")
        A("| 口径 | 分子/分母 | 比率 | 说明 |")
        A("|---|---|---|---|")
        A(f"| 上界（含碎片，主口径） | {fd['上界分子分母']} | **{fd['上界（含碎片，主口径）']:.4f}** | "
          f"碎片 = 抽取残片（截断/句片段/体裁词），不是公司名 |")
        A(f"| 剔碎片后 | {fd['剔碎片后分子分母']} | **{fd['剔碎片后']:.4f}** | 分子分母同剔碎片 |")
        A(f"| 修正前对照（v4 交付口径） | {fd['修正前对照（v4 交付口径）']['含碎片分子分母']} | "
          f"{fd['修正前对照（v4 交付口径）']['含碎片上界']:.4f} | 复核口径剔 12 碎片后 = "
          f"{fd['修正前对照（v4 交付口径）']['剔碎片后（复核清单 12 碎片）']:.4f} |")
        A("")
        A(f"**本轮实测三分类**（unmapped 唯一名 {sum([fd['碎片'], fd['名称漂移'], fd.get(RESIDUAL_CLS, 0)])} 个）："
          f"碎片 **{fd['碎片']}** / 名称漂移 **{fd['名称漂移']}** / 残差桶（未过前三类判据） **{fd.get(RESIDUAL_CLS)}**。")
        for k in ("碎片", "名称漂移", RESIDUAL_CLS):
            items = fd["明细"].get(k) or []
            if items:
                A(f"- {k}（{len(items)}）：" + "、".join(f"{x['name']}（{x['evidence']}）" for x in items[:12])
                  + ("…" if len(items) > 12 else ""))
        A("")
        A("（分类为**规则判读**口径：逐名证据见 JSON `failure_decomposition.明细`；主口径失败率不依赖该分类。"
          "已知噪声：2 字同头判据已排除地名前缀（PLACE_PREFIX），仍可能有少量子公司/全称形态落入"
          "「残差桶（未过前三类判据）」（registry 1.5.17 修 2 改名；原类名「非 A 股实体」为兜底断言），"
          "属描述性口径的已知不确定性。）")
        A("")
    ch = r["rates"].get("extraction_channels")
    if ch:
        A("提取通道计数（修 3）：" + "；".join(f"{k} = {v}" for k, v in ch.items()) + "。")
        A("")
    A("## 2. 锚读数（lift）")
    A("")
    A(f"口径（P0 登记）：lift = P(自报同行 ∈ 网络对手 | 自报同行) ÷ P(随机 universe 公司 ∈ 网络对手)，"
      f"分母基线 = 该网络年无向对密度 d = E/(N(N−1)/2)。")
    A("")
    A("| 读数 | 重合率（原始） | 边对数/总对数 | 基线密度 d | lift |")
    A("|---|---|---|---|---|")
    A(f"| 主口径（全部自报同行对） | **{an['overlap_share_raw']:.6f}** | {an['pair_edge']}/{an['pair_total']} | "
      f"{an['base_density']:.6f} | **{an['lift']:.4f}** |")
    A(f"| 诊断·限双方均在网 | {an['diagnostic_node_restricted']['overlap_share_raw']:.6f} | "
      f"{an['diagnostic_node_restricted']['pair_edge']}/{an['diagnostic_node_restricted']['pair_total']} | "
      f"{an['base_density']:.6f} | {an['diagnostic_node_restricted']['lift']:.4f} |")
    A("")
    A(f"网络：文本年 {nb['text_year']}（`{nb['version']}`）、节点 N = {nb['N_nodes']}、无向边 E = {nb['E_undirected']}。")
    A(f"逐公司重合率分布（在网且已有映射同行的 {ov['n_firms_in_network_with_peers']} 家）："
      f"均值 {ov['mean']:.4f} / 中位 {ov['median']:.4f} / 最小 {ov['min']:.4f} / 最大 {ov['max']:.4f}；"
      f"零重合 {ov['n_zero_overlap']} 家、≥50% {ov['n_ge_50pct']} 家、≥90% {ov['n_ge_90pct']} 家。")
    A("")
    ep = r.get("extraction_pollution")
    if ep:
        A("### 2.1 抽取污染登记与敏感性（修 1）")
        A("")
        A(f"**问题**：{ep['issue']}")
        A("")
        A("**污染样例**（独立复核 14 例）：")
        for x in ep["examples"]:
            A(f"- {x}")
        A("")
        A(f"**修法**：{ep['fix']}")
        A("")
        A(f"**残余**：{ep['residual']}")
        A("")
        A("**敏感性（修正前 / 复核口径 / 修正后 三套读数，同一网络版本同一基线）**：")
        A("")
        A("| 口径 | 重合率（主） | 边对/总对 | lift（主） | 诊断·限在网 | lift（诊断） |")
        A("|---|---|---|---|---|---|")
        for k, v in ep["sensitivity"].items():
            A(f"| {k} | {v['overlap_share']:.6f} | {v['pair_edge']}/{v['pair_total']} | "
              f"{v['lift']:.4f} | {v['diag_overlap_share']:.6f}（{v['diag_pair_edge']}/{v['diag_pair_total']}） | "
              f"{v['diag_lift']:.4f} |")
        A("")
    dn = r.get("direction_note")
    if dn:
        A("### 2.2 下界与方向性说明（修 4）")
        A("")
        A(f"- {dn['coverage_lower_bound']}")
        A(f"- {dn['lift_direction_unproven']}")
        A(f"- {dn['probe']}")
        dc = an.get("direction_check") or {}
        if dc:
            rr = dc["rejected_pairs_edge_rate"]
            rr_s = "n/a" if rr is None else f"{rr:.4f}"
            A(f"- 内联方向性检查：被剔名-码对（双方均在网）{dc['rejected_pairs_edge']}/"
              f"{dc['rejected_pairs_in_network']} = {rr_s}"
              f"（保留集诊断口径 {dc['kept_pairs_edge_rate_diag']:.4f}，"
              f"{dc['kept_pairs_edge']}/{dc['kept_pairs_in_network']}）"
              f" → 剔除的名-码对中含真边，方向与「漏抽压低 lift」相反 → lift 方向未证。")
        A("")
    A("**角色限定**：本表为**外部锚（锚 1）试点读数**，只回答「网络边是否落在公司自报同行上」，"
      "不进入面板/回归、不改主口径；**单年试点不构成「外部锚条件满足」**（全量见 §5）。")
    A("")
    A("## 3. 抽样口径与解析修复轮次（停机 → 先修解析 → 复测）")
    A("")
    A(f"- 关键词集合 = {' / '.join(sc['keyword_set'])}；定位窗口 = hit−{sc['window'][0]}..hit+{sc['window'][1]} 字。")
    A(f"- 抽取规则 = {' / '.join(sc['extraction_rules'])}；映射模式 = {' / '.join(sc['mapping_modes'])}"
      f"（**仅精确匹配 + 公司后缀归一化 + 代码锚定，无模糊匹配、无人工补录**）。")
    A(f"- 名称字典来源：项目 firm 名单（`firm_id_map.csv` {sc['name_dict']['firm_universe']} 只，"
      f"**去 ST/*ST 前缀标记建键**）+ `a_stocks.csv` + `name_history_*.csv` + 年报头部「简称+代码」配对"
      f"（标签/顺序/冒号变体、头部窗口 24000 字、同头拼接去重）+ 更名披露（「证券简称由 X 变更为 Y」）"
      f"+ 释义缩写（「X 指 Y（…公司）」，后缀一致过滤）+ 东财曾用名缓存；"
      f"简称键 {sc['name_dict']['short_keys']}、全称键 {sc['name_dict']['full_keys']}。")
    A("")
    A("| 轮次 | 映射失败率 | 未映射/唯一名称 | 覆盖率 | lift | 修法 | 证据日志 |")
    A("|---|---|---|---|---|---|---|")
    for it in r["rates"]["parse_iterations"]:
        A(f"| {it['round']} | {it['mapping_failure_rate']:.4f} | {it['n_unmapped']}/{it['n_unique']} | "
          f"{it['coverage_rate']:.4f} | {it['lift']:.4f} | {it['note']} | `{it['log']}` |")
    A("")
    A("## 4. 失败案例样本（≤10 条）")
    A("")
    if r["failure_cases_sample"]:
        A("| 类型 | 代码 | 名称 | 细节 |")
        A("|---|---|---|---|")
        for c in r["failure_cases_sample"]:
            det = c.get("detail") or c.get("unmapped") or c.get("peers") or ""
            if isinstance(det, list):
                det = "、".join(str(x) for x in det)
            A(f"| {c['case']} | {c.get('code','')} | {c.get('name','')} | {det} |")
    else:
        A("（无）")
    A("")
    A("未映射名称 Top（跨公司出现次数）：" +
      "、".join(f"{x['name']}×{x['n_firms']}" for x in r["unmapped_top30"][:15]))
    A("")
    A("## 5. 下一步全量可行性判断（路线 B）")
    A("")
    n_nosection = r["rates"]["coverage_breakdown"].get("no_section（解析未定位，计入解析分母）", 0)
    cid_docs = [x for x in r["per_firm"] if str(x.get("channel", "")).startswith("pdfminer+cid_gid")]
    A(f"- **解析层**：{r['rates']['parse_success_rate']:.1%} 定位成功率。失败分解（修 3 单列）："
      f"① **无文本层扫描件** 2 份（300760/300748，pymupdf 与 pdfminer 均无文本层 → 需 OCR，属路线 B 前置）；"
      f"② **文本层 CID 编码损坏** {len(cid_docs)} 份（300737/603059/603301：无 ToUnicode CMap，本轮经 "
      f"pdfminer.six + SimSun GID 反查备选通道**已恢复**，并已进入抽取/映射）；"
      f"③ 其余 {max(0, n_nosection - len(cid_docs))} 份为**非标准/缺失章节标题**（4 关键词命中 = 0）。"
      "全量可复用同一脚本（`--year` 参数化），成本 = 每年下载 + 解析（本试点 105 份约 4 分钟，含备选通道）。")
    fd2 = r["rates"].get("failure_decomposition") or {}
    A(f"- **映射层**：主口径失败率 {r['rates']['mapping_failure_rate']:.1%}（上界口径，含碎片）"
      f"；**实测三分类** = 碎片 {fd2.get('碎片', '?')} / 名称漂移 {fd2.get('名称漂移', '?')} / "
      f"残差桶（未过前三类判据）{fd2.get(RESIDUAL_CLS, '?')}（逐名证据见 §1.1）。该桶来源需独立判读"
      "（未上市同业 / 境外上市 / 港股），属口径内不可映射；名称漂移由年报头部「简称+代码」（标签/顺序/冒号变体、"
      "去 ST 前缀）、更名披露与释义缩写四条**代码锚定**来源覆盖（本轮补齐 5 例复核确认的漏配）。")
    A(f"- **覆盖率**：{r['rates']['coverage_rate']:.1%}（**下界**；分母含 2 份无文本层 + "
      f"{max(0, n_nosection - len(cid_docs))} 份非标准标题未定位 + "
      f"{r['rates']['coverage_breakdown'].get('located_but_no_mapped_peer', 0)} 份抽到名单但无可映射同行）。"
      "全量若保持同量级，锚表可覆盖网络节点内约 4/5 的 IPO 公司。")
    A("- **lift 量级**：单年试点 lift ≈ "
      f"{r['anchor_readout']['lift']:.1f}（对照：分析师共同覆盖锚池化 5.39；同行业 lift 13–22）。"
      "量级与「自报同行更可能落在网内边」一致，但**单年、单网络版本，不能外推为全期结论**；"
      "且**方向未证**（§2.2）：漏抽与污染剔除两侧偏置方向不一致。")
    A("- **建议**：① 全量前补 OCR 通道处理无文本层扫描件（2 份）；② 逐年重复本脚本并汇总为 "
      "`s3_network_prospectus_anchor.json`；③ **在完成全量（≥3 年）之前，论文仍不得写 pre-validated**；"
      "④ 全量轮补「表格列错行 → 码-名配对校验」（本轮已知残余，见 §6）。")
    A("")
    A("## 6. 已知限制")
    A("")
    for x in r.get("known_limitations", []):
        A(f"- {x}")
    A("")
    A("## 7. 复现与产物版本关系")
    A("")
    A("```bash")
    A(f"python scripts/fetch_cninfo_prospectus.py --ipo-year {sc['pilot_year']} --workers 3")
    A(f"python scripts/network_prospectus_anchor.py --year {sc['pilot_year']}")
    A("```")
    A("")
    A("- **明细 CSV 版本关系（修 5）**：本报告同目录 `s3_network_prospectus_pilot_2018_detail.csv` 为**权威交付版**"
      "（脚本 `--detail-csv` 直出，105 行）。修正轮清理了中间产物 `artifacts/tmp/tier2_prosp/pilot_2018_detail.csv`"
      "（v4 期中间件，与交付版有 **9 行 × 6 字段 = 54 处字段填充差异**：`no_section` 行在中间件里留空、"
      "交付版填 0/True；sha256 = 45a61e31… 已留档 `artifacts/tmp/tier2_prosp_fix/f5_tmp_evidence.json`）。"
      "**以交付版为准，中间件不再保留、不得引用。**")
    A("- **tmp 大件清理（修 5）**：`artifacts/tmp/tier2_prosp/` 下 2 份探针 PDF（300748 165 MB / 300760 31 MB，"
      "用于「无文本层扫描件」实测：413 页 948 字 / 620 页 2,480 字）已删除，其 sha256、字节数与页/字计数留档于"
      " `artifacts/tmp/tier2_prosp_fix/f5_tmp_evidence.json`（PDF 可经 P1 下载器重取；下载日志 "
      "`download_log.csv` 已登记 `all_candidates_failed` + `extract_fail`）。")
    A("- **CID 恢复缓存（修 3）**：`artifacts/tmp/tier2_prosp/cid_recover/<code>.{txt,key}`（3 份解码文本 + "
      "PDF 字节数键；删缓存后可由 `pdfminer.six + SimSun GID 反查`确定性重建，需 `C:/Windows/Fonts/simsun.ttc`）。")
    A("- **修正轮探针与证据**：`artifacts/tmp/tier2_prosp_fix/`（候选 dump、R3 规则变体网格、CID 恢复探针、"
      "方向性检查、tmp 清理证据、四审计与自检日志）。")
    A("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

def _f(x, spec=".4f"):
    """None 安全的数值格式化（边界年/空集情形）。"""
    if x is None:
        return "n/a"
    try:
        return f"{x:{spec}}"
    except (TypeError, ValueError):
        return str(x)

def write_anchor_md(r, path):
    """全量锚报告人读版：逐年表 + 池化 + 边界年 + 外部锚判定 + 限制 + 试点对照。"""
    sc, ra, ex = r["scope"], r["anchor_readout"], r["external_anchor"]
    ov = r["rates"]["overall"]
    fd = r["rates"]["failure_decomposition_pooled"]
    L = []
    A = L.append
    A("# 第二梯队 #9 路线 B 全量：招股书「同行业可比公司」自报锚（锚 1 / plan §8 B″11-7）")
    A("")
    A(f"- 运行时间（UTC）：{r['run_at']}；口径参数：`{r['parameter']}`")
    fr2 = r.get("fix_round2") or {}
    A(f"- 范围：IPO 年 **{sc['ipo_years'][0]}–{sc['ipo_years'][-1]} 全量不抽样**；cohort 合计 **{sc['cohort_n']}**"
      f"（universe = `{sc['universe']}`）；四项定义 / 停机线 / 匹配口径逐字沿用试点登记值；"
      f"解析层为 **registry {r.get('fix_round', {}).get('registry_version', '1.5.15')} H1/M1/M2 修复轮**后的读数；"
      f"本轮为 **registry {fr2.get('registry_version', '1.5.17')} 修复轮二（修 1–7）**读数"
      f"（上一轮 1.5.16 = 本轮修复前基线，见 §0）。")
    if fr2:
        A(f"- **报告头登记版本（修 5）**：本报告 run 时的实际 registry 版本 = **registry "
          f"{fr2.get('registry_version', '1.5.17')}**（**非** 1.5.15 —— 前版报告头写 1.5.15 为登记文本滞后，已改）；"
          f"`name_dict.short_keys` **净减 1,297**（H1 错切键移除，13,412→12,115；非人工删除）。")
    A(f"- 网络版本：逐 IPO 年 Y 用其 `network_version_used`（y_target = min(Y+2, 2023)；v1_2023 不存在 → "
      f"fallback 最近可用 v1_2022）；**主读数年 = {ra['pooled_main_years_2015_2020']['years']}**；"
      f"边界年子集 = {list(ra['boundary_subset'].keys())}（单列、不混入主读数）。")
    A("")
    # ---- 修复轮与前置/后对照（registry 1.5.15）----
    pre = r.get("sensitivity_pre_fix")
    pre2 = r.get("sensitivity_pre_round2")
    fr = r.get("fix_round") or {}
    if fr:
        A("## 0. 修复轮与修复前/后对照（三轮读数：v1 → 上一轮 H1/M1/M2 → 本轮修 1–7）")
        A("")
        A(f"- 登记：{fr.get('change', '')}")
        A(f"- **判据零改动**：{fr.get('judgment_unchanged', '')}")
        A(f"- 归因口径（additive）：{fr.get('attribution_additive', '')}")
        if fr2:
            A(f"- **本轮（registry {fr2.get('registry_version', '1.5.17')}）登记**：{fr2.get('change', '')}")
            A(f"- **本轮判据零改动**：{fr2.get('judgment_unchanged', '')}")
            A(f"- 重抽口径：{fr2.get('re_extract_scope', '')}")
        A("")
        if pre and not pre.get("error"):
            A(f"v1 基线来源：`{pre['source']}`（run_at {pre['run_at_before_fix']}）。{pre['note']}")
            if pre2 and not pre2.get("error"):
                A("")
                A(f"**本轮修复前基线（上一轮 H1/M1/M2 修复后，registry 1.5.16）**：`{pre2['source']}`"
                  f"（run_at {pre2['run_at_before_fix2']}）。{pre2['note']}")
            A("")
            A("| 指标 | v1 基线（1.5.13/1.5.14） | 上一轮修复后（1.5.16） | 本轮修复后（1.5.17） | 停机线 | 判定 |")
            A("|---|---|---|---|---|---|")
            p2c = (pre2 or {}).get("cls4") or {}
            A(f"| 解析成功率 | {_f(pre['parse_success_rate'])}（{pre['parse_n']}） | "
              f"{_f((pre2 or {}).get('parse_success_rate'))}（{(pre2 or {}).get('parse_n', '')}） | "
              f"**{_f(ov['parse_success_rate'])}**（{ov['parsed']}/{ov['downloaded']}） | ≥0.80 | "
              f"{'PASS' if ov['parse_success_rate'] >= 0.80 else 'FAIL'} |")
            A(f"| 映射失败率（上界，主口径） | {_f(pre['mapping_failure_rate_upper'])}（{pre['mapping_failure_n']}） | "
              f"{_f((pre2 or {}).get('mapping_failure_rate_upper'))}（{(pre2 or {}).get('mapping_failure_n', '')}） | "
              f"**{_f(fd['上界（含碎片，主口径）'])}**（{fd['上界分子分母']}） | ≤0.20 | "
              f"{'PASS' if fd['上界（含碎片，主口径）'] <= 0.20 else 'FAIL'} |")
            A(f"| 映射失败率·剔碎片与噪声后（诊断） | {_f(pre['mapping_failure_ex_frag'])}（{pre['mapping_failure_ex_frag_n']}） | "
              f"{_f((pre2 or {}).get('mapping_failure_ex_frag_noise'))}（{(pre2 or {}).get('mapping_failure_ex_frag_noise_n', '')}） | "
              f"{_f(fd['剔碎片与噪声后'])}（{fd['剔碎片与噪声后分子分母']}） | —（诊断列） | — |")
            A(f"| 覆盖率 | {_f(pre['coverage_rate'])}（{pre['coverage_n']}） | "
              f"{_f((pre2 or {}).get('coverage_rate'))}（{(pre2 or {}).get('coverage_n', '')}） | "
              f"**{_f(ov['coverage_rate'])}**（{ov['n_covered']}/{ov['cohort']}） | ≥0.50 | "
              f"{'PASS' if ov['coverage_rate'] >= 0.50 else 'FAIL'} |")
            A(f"| 池化 lift（主读数年 2015–2020） | {_f(pre['pooled_lift'])}（{pre['pooled_pair']}） | "
              f"{_f((pre2 or {}).get('pooled_lift'))}（{(pre2 or {}).get('pooled_pair', '')}） | "
              f"**{_f(ra['pooled_main_years_2015_2020']['lift'])}**"
              f"（{ra['pooled_main_years_2015_2020']['pair_edge']}/{ra['pooled_main_years_2015_2020']['pair_total']}） | —（读数） | — |")
            A(f"| 四分类（唯一名） | 碎片 {(pre.get('cls3') or {}).get('碎片')} / 名称漂移 {(pre.get('cls3') or {}).get('名称漂移')} / "
              f"非A股实体（旧名）{(pre.get('cls3') or {}).get('非 A 股实体')} | 碎片 {p2c.get('碎片')} / 名称漂移 {p2c.get('名称漂移')} / "
              f"抽取噪声 {p2c.get('抽取噪声')} / 残差桶 {p2c.get(RESIDUAL_CLS)} | 碎片 {fd['碎片']} / 名称漂移 {fd['名称漂移']} / "
              f"抽取噪声 {fd['抽取噪声']} / 残差桶 {fd.get(RESIDUAL_CLS)} | —（描述性） | — |")
            A(f"| 外部锚条件（`external_anchor_satisfied`） | {pre['external_anchor_satisfied']} | "
              f"{(pre2 or {}).get('external_anchor_satisfied')} | {ex['external_anchor_satisfied']} | 三项全过 ∧ lift 齐全 | "
              f"{'满足' if ex['external_anchor_satisfied'] else '未满足'} |")
            A("")
            A("**逐年对照（v1 → 上一轮 → 本轮）**")
            A("")
            A("| IPO 年 | 解析率 | 覆盖率 | 映射失败率 | lift |")
            A("|---|---|---|---|---|")
            for y in map(str, sc["ipo_years"]):
                a = pre["per_year"].get(y, {})
                b = ((pre2 or {}).get("per_year") or {}).get(y, {})
                by = r["rates"]["per_year"][y]
                ly = ra["per_year"][y]
                A(f"| {y} | {_f(a.get('parse_success_rate'))} → {_f(b.get('parse_success_rate'))} → {_f(by['parse_success_rate'])}"
                  f" | {_f(a.get('coverage_rate'))} → {_f(b.get('coverage_rate'))} → {_f(by['coverage_rate'])}"
                  f" | {_f(a.get('mapping_failure_rate'))} → {_f(b.get('mapping_failure_rate'))} → "
                  f"{_f(r['rates']['failure_decomposition_per_year'][y]['上界（含碎片，主口径）'])}"
                  f" | {_f(a.get('lift'))} → {_f(b.get('lift'))} → {_f(ly['lift'])} |")
            A("")
        elif pre and pre.get("error"):
            A(f"（修复前基线读取失败：{pre['error']}；来源 `{pre.get('source', '')}`）")
            A("")
    A("## 1. 四项指标（全量）")
    A("")
    A("| 指标 | 实测 | 分子/分母 | 停机线 | 判定 |")
    A("|---|---|---|---|---|")
    A(f"| 解析成功率 | **{ov['parse_success_rate']:.4f}** | {ov['parsed']}/{ov['downloaded']}（下载成功份数）"
      f" | ≥0.80 | {'PASS' if ov['parse_success_rate'] >= 0.80 else 'FAIL'} |")
    A(f"| 映射失败率（上界，主口径） | **{fd['上界（含碎片，主口径）']:.4f}** | {fd['上界分子分母']}（跨年去重唯一名）"
      f" | ≤0.20 | {'PASS' if fd['上界（含碎片，主口径）'] <= 0.20 else 'FAIL'} |")
    A(f"| 覆盖率 | **{ov['coverage_rate']:.4f}** | {ov['n_covered']}/{ov['cohort']}"
      f"（2015–2023 cohort ∩ 项目 universe） | ≥0.50 | {'PASS' if ov['coverage_rate'] >= 0.50 else 'FAIL'} |")
    A("")
    A(f"**停机判定 = {'触发（' + '；'.join(ex['stop_reasons']) + '）' if ex['stop_triggered'] else '未触发（三项全过）'}**；"
      f"**外部锚（锚 1）条件 = {'已满足（external_anchor_satisfied = true）' if ex['external_anchor_satisfied'] else '未满足'}**"
      f"（判据与 lift 报告完备性见 §5）。")
    A(f"- **网络节点两口径（修 6）**：`n_network_nodes` = **{ov.get('n_network_nodes')}**（**located 口径**：已定位件中"
      f"在该网络版本为节点的家数）；`n_network_nodes_member_scope` = **{ov.get('n_network_nodes_member_scope')}**"
      f"（**预登记会员口径**：cohort 中 firm_id ∈ 该网络版本节点集的家数，含未下载/未定位件；预登记轮登记值 "
      f"= 1,537/1,592 = 96.5%）——两口径并报、不得混用。")
    A("- **`first_hit_page` 语义（修 4）**：非 pymupdf 通道（CID 确定性还原 / OCR 回退）置 `not_applicable_channel`"
      "（替换后文本的字符偏移与 pymupdf 页码不对应）；pymupdf 通道报真实页码。")
    A("")
    A("### 1.1 映射失败率口径分列（上界 / 剔碎片 / 剔碎片与噪声 / 四分类）")
    A("")
    A("| 口径 | 唯一名 | 未映射 | 上界（含碎片） | 剔碎片后 | 剔碎片与噪声后 | 碎片 | 名称漂移 | 抽取噪声 | 残差桶（未过前三类判据） |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for key, lab in (("failure_decomposition_pooled", "**全量池化（主口径）**"),
                     ("failure_decomposition_main_years", "主读数年池化（2015–2020，诊断）"),
                     ("failure_decomposition_boundary", "边界年子集（2021–2023，单列）")):
        d = r["rates"][key]
        A(f"| {lab} | {d['n_unique_names']} | {d['n_unmapped']} | **{_f(d['上界（含碎片，主口径）'])}**"
          f" | {_f(d['剔碎片后'])} | {_f(d['剔碎片与噪声后'])} | {d['碎片']} | {d['名称漂移']}"
          f" | {d['抽取噪声']} | {d.get(RESIDUAL_CLS)} |")
    A("")
    A("**逐年映射失败率（年内去重，诊断分列）**：")
    A("")
    A("| IPO 年 | 唯一名 | 未映射 | 上界 | 剔碎片后 | 剔碎片与噪声后 | 碎片 | 名称漂移 | 抽取噪声 | 残差桶（未过前三类判据） |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for y in map(str, sc["ipo_years"]):
        d = r["rates"]["failure_decomposition_per_year"][y]
        A(f"| {y} | {d['n_unique_names']} | {d['n_unmapped']} | {_f(d['上界（含碎片，主口径）'])}"
          f" | {_f(d['剔碎片后'])} | {_f(d['剔碎片与噪声后'])} | {d['碎片']} | {d['名称漂移']}"
          f" | {d['抽取噪声']} | {d.get(RESIDUAL_CLS)} |")
    A("")
    A(f"（分类 = 规则判读口径（逐名证据见 JSON `failure_decomposition_*.明细`）；主口径失败率不依赖该分类。"
      f"第 4 类「抽取噪声」= registry 1.5.15 ④ 新增、**registry 1.5.17 修 3 改判据**（= **整名 ≈ 通用词**"
      f"（整名 ∈ 词表 或 ≤{GEN_SUB_MAXLEN} 字且含词表词）+ **公司后缀豁免**：公司/集团/股份/有限/企业 收尾或含 股份/集团）；"
      f"**「剔碎片后」列口径未改**。全量池化未映射唯一名 {fd['n_unmapped']} 个：碎片 {fd['碎片']} / 名称漂移 {fd['名称漂移']} / "
      f"抽取噪声 {fd['抽取噪声']} / **残差桶（未过前三类判据）** {fd.get(RESIDUAL_CLS)} —— 残差桶为**兜底 return**，"
      f"其来源需独立判读，**不得**以该桶断言「非 A 股实体」（修 2 改名；原类名/字段名 `非 A 股实体`/`non_a_share_split`）。）")
    A("")
    for k in ("名称漂移", "抽取噪声", RESIDUAL_CLS):
        items = fd["明细"].get(k) or []
        if items:
            A(f"- {k}（{len(items)}）：" + "、".join(x["name"] for x in items[:40]) + ("…" if len(items) > 40 else ""))
    A("")
    A("### 1.2 失败构成诊断（描述性；不改主口径、不参与门槛判定）")
    A("")
    fdg = r.get("failure_diagnostics")
    if fdg:
        A(fdg["note"])
        A("")
        _rb = fdg.get("residual_bucket_audit") or {}
        _sa = _rb.get("sampling_audit") or {}
        A(f"- unmapped 唯一名 {fdg['n_unmapped_unique']} 个；四分类（碎片 "
          f"{fd['碎片']} / 名称漂移 {fd['名称漂移']} / 抽取噪声 {fd['抽取噪声']} / "
          f"**残差桶（未过前三类判据）** {fd.get(RESIDUAL_CLS)}）。")
        A(f"- **残差桶（修 2，{_rb.get('bucket_n', fd.get(RESIDUAL_CLS))} 个）需独立判读**：抽样审计（n="
          f"{_sa.get('n_sampled')}）显示 真非上市/境外 {_sa.get('真非上市/境外')}"
          f"（{_sa.get('share', {}).get('真非上市/境外', 0) * 100:.1f}%）/ 抽取伪影 {_sa.get('抽取伪影')}"
          f"（{_sa.get('share', {}).get('抽取伪影', 0) * 100:.1f}%）/ A 股名漂移 {_sa.get('A股名漂移')}"
          f"（{_sa.get('share', {}).get('A股名漂移', 0) * 100:.1f}%）→ **不是单一来源**。")
        _qd = _rb.get("quantified_decomposition") or {}
        _lc = _qd.get("loosest_correction") or {}
        A(f"- **复核量化分解**：{_qd.get('note', '')}")
        A(f"- **最宽松修正（同剔全部抽取伪影 844；口径 = 复核的修复前分母 5818）**："
          f"失败率 = {_lc.get('numerator_denominator')} = **{_lc.get('rate'):.4f}**"
          f"（复核载 {_lc.get('review_value')}）**仍 > 0.20** → **停机不解除**、外部锚条件仍不满足（未放宽任何门槛）；"
          f"（按本轮修复后分母重算同口径剔伪影读数 = (2266−844)/(6175−844)，仍 > 0.20，判定不变。）"
          if _lc.get('rate') is not None else "- （复核量化分解缺省）")
        A(f"- 读法：{fdg['outcome_reading']}")
        A("")
        A("| # | 未映射名（按文档频次） | 文档数 | 四分类 | 出现年 | 证据 |")
        A("|---|---|---|---|---|---|")
        for i, x in enumerate(fdg["top_unmapped_by_docfreq"][:25], 1):
            A(f"| {i} | {x['name']} | {x['n_docs']} | {x['cls']} | {','.join(map(str, x['years']))}"
              f" | {x['evidence']} |")
        A("")
    A("## 2. 逐年表（cohort / 下载 / 解析 / 覆盖率 / 映射失败 / lift + CI）")
    A("")
    A("| IPO 年 | cohort | 下载 | 解析 | 解析率 | 覆盖率 | 映射失败（上界） | 网络版本 | 对数/命中 | 重合率 | lift | lift CI95 | 诊断·限在网 lift | 边界年 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for y in map(str, sc["ipo_years"]):
        py = r["rates"]["per_year"][y]
        ly = ra["per_year"][y]
        lift_s = "n/a" if ly["lift"] is None else f"{ly['lift']:.4f}"
        ci = ly.get("lift_ci95") or [None, None]
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "n/a"
        dg = "n/a" if ly["diag_lift"] is None else f"{ly['diag_lift']:.4f}"
        mf = r["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"]
        share_s = "n/a" if ly["overlap_share"] is None else f"{ly['overlap_share']:.6f}"
        A(f"| {y} | {py['cohort']} | {py['downloaded']} | {py['parsed']} | {py['parse_success_rate']:.4f}"
          f" | **{py['coverage_rate']:.4f}**（{py['n_covered']}） | {_f(mf)} | `{ly['network_version_used']}`"
          f" | {ly['pair_edge']}/{ly['pair_total']} | {share_s}"
          f" | **{lift_s}** | {ci_s} | {dg} | {'是' if py['boundary_year_subset'] else '否'} |")
    A("")
    A(f"- 合计：cohort **{ov['cohort']}** / 下载 {ov['downloaded']} / 解析 {ov['parsed']}（{ov['parse_success_rate']:.4f}）"
      f"/ 覆盖 {ov['n_covered']}（{ov['coverage_rate']:.4f}）/ no_section {ov['no_section']} / no_file {ov['no_file']}"
      f" / parse_error {ov['parse_error']}。")
    A("")
    A("## 3. 锚读数（lift）")
    A("")
    A(f"口径：{ra['definition']}")
    A("")
    pm = ra["pooled_main_years_2015_2020"]
    pa = ra["pooled_including_boundary"]
    A("| 读数 | 对数/命中 | 重合率 | 池化密度 d_w | lift | lift CI95 | 诊断·限双方在网 lift |")
    A("|---|---|---|---|---|---|---|")
    A(f"| **池化（主读数年 {pm['years']}）** | {pm['pair_edge']}/{pm['pair_total']} | **{_f(pm['overlap_share'], '.6f')}**"
      f" | {_f(pm['pooled_density_dw'], '.6f')} | **{_f(pm['lift'])}** | "
      f"[{_f(pm['lift_ci95'][0], '.2f')}, {_f(pm['lift_ci95'][1], '.2f')}] | {_f(pm['diag_lift'])} |")
    A(f"| 池化（含边界年，对照） | {pa['pair_edge']}/{pa['pair_total']} | {_f(pa['overlap_share'], '.6f')}"
      f" | {_f(pa['pooled_density_dw'], '.6f')} | {_f(pa['lift'])} | — | {_f(pa['diag_lift'])} |")
    A("")
    ovs = ra["per_firm_overlap_stats"]
    A(f"逐公司重合率（在网且有映射同行 {ovs['n_firms_in_network_with_peers']} 家）：均值 {ovs['mean']:.4f}；"
      f"零重合 {ovs['n_zero_overlap']} 家、≥50% {ovs['n_ge_50pct']} 家。")
    A("")
    A("### 3.1 边界年子集（{2021, 2022, 2023}，单列、不混入主读数）")
    A("")
    A("| IPO 年 | cohort | 下载 | 解析 | 网络版本 | 该版本中节点数 | 对数 | lift |")
    A("|---|---|---|---|---|---|---|---|")
    for y, b in ra["boundary_subset"].items():
        lift_s = "n/a" if b["lift"] is None else f"{b['lift']:.4f}"
        A(f"| {y} | {b['cohort']} | {b['downloaded']} | {b['parsed']} | `{b['network_version_used']}`"
          f" | **{b['n_network_nodes']}** | {b['pair_total']} | {lift_s} |")
    A("")
    A("边界年处理：y_target = min(Y+2, 2023) 不可得（v1_2023 不存在）→ fallback 最近可用网络版本 v1_2022；"
      "三 cohort 在该版本中节点数 = 0（冻结样本筛选「上市不足 2 年」）→ 锚读数退化（无对）；"
      "下载/解析/覆盖计数全量纳入，**未静默剔除**。")
    A("")
    dc = ra["direction_check"]
    rr = dc["rejected_pairs_edge_rate"]
    A(f"**方向性说明（沿用试点「方向未证」的如实表述）**：被剔名-码对（双方均在网）{dc['rejected_pairs_edge']}/"
      f"{dc['rejected_pairs_in_network']} = {'n/a' if rr is None else f'{rr:.4f}'}"
      f"（保留集诊断口径 {_f(dc['kept_pairs_edge_rate_diag'])}）→ 剔除/漏抽两侧偏置方向不一致 → "
      f"**lift 方向未证**（不宣称下界、不宣称上界）。")
    A("")
    A("## 4. 通道与失败披露（不得静默剔除）")
    A("")
    cr = r.get("ingest_crosscheck") or {}
    A(f"**与全量第一轮 ingest 明细逐份对账**：比较 {cr.get('n_compared', 0)} 份；**status 不一致 "
      f"{len(cr.get('status_mismatch') or [])} 份**（口径未漂移）；**channel 标签不一致 "
      f"{len(cr.get('channel_mismatch') or [])} 份**——全部为 OCR/CID 件的**运行元数据差**"
      f"（ingest 轮记 `render/ocr` 耗时、本轮缓存命中记 `（缓存命中）`），**终态通道一致**"
      f"（逐条见 JSON `ingest_crosscheck.channel_mismatch`）。")
    A("")
    ch = r["channel_disclosure"]
    A(f"非 pymupdf 通道 **{len(ch)} 件**（OCR 回退 / CID 确定性还原；逐件 channel 见 JSON `channel_disclosure`）：")
    A("")
    A("| IPO 年 | code | 通道 | 页数 | 字数 | 关键词命中 | 解析状态 |")
    A("|---|---|---|---|---|---|---|")
    for x in ch:
        A(f"| {x['year']} | {x['code']} | `{x['channel']}` | {x['pages']} | {x['chars']} | {x['kw_hits']} | {x['status']} |")
    A("")
    dl = r["download_failures"]
    A(f"下载失败 **{dl['n_failed']} 件**（status 概览 = {dl['by_status']}）；逐条：")
    A("")
    A("| IPO 年 | code | download_log 末次 status |")
    A("|---|---|---|")
    for x in dl["items"]:
        A(f"| {x['year']} | {x['code']} | `{x['status']}` |")
    A("")
    A("## 5. 外部锚（锚 1）条件判定（机械、不得放宽）")
    A("")
    A("| 判据 | 门槛 | 实测 | 判定 |")
    A("|---|---|---|---|")
    for k, v in ex["criteria"].items():
        val = v["value"]
        A(f"| {k} | {'≥0.80' if '解析' in k else ('≤0.20' if '映射' in k else '≥0.50')} | {_f(val)}"
          f"（{v.get('scope', '')}） | {'PASS' if v['pass'] else 'FAIL'} |")
    A(f"| lift 报告完备性 | 逐年 + 池化 + 边界年单列 + 诊断口径 | {'齐全' if ex['lift_reports_complete'] else '不齐全'} | "
      f"{'PASS' if ex['lift_reports_complete'] else 'FAIL'} |")
    A("")
    A(f"**判定：外部锚（锚 1）条件 = {'已满足' if ex['external_anchor_satisfied'] else '未满足'}**"
      f"（`external_anchor_satisfied = {str(ex['external_anchor_satisfied']).lower()}`）。")
    if ex["external_anchor_satisfied"]:
        A("")
        A("据此登记：**「pre-validated / 已验证」措辞限制在本条件（外部锚条件已满足）下解除**——"
          "解除范围 = 论文/材料中针对**外生锚定证据**的「pre-validated（预先验证过的）」表述，"
          "条件 = 本报告登记的四项指标与 lift 读数（registry change_log 同文登记）。"
          "**但该解除与 #10 的「AI 独立判读（三代理，非人工）」证据层级限定并列存在、互不替代**："
          "涉及**「边级语义」（网络中的边 = 真实竞争关系）**的表述仍受 #10 证据层级与门槛判定"
          "（`TRIGGER_NETWORK_RECHECK`）约束，不得据本报告写「网络已验证为真实竞争关系」。")
    else:
        A("")
        A("据此登记：**外部锚（锚 1）条件未满足**（`external_anchor_satisfied = false`），卡点 = "
          + "；".join(ex["stop_reasons"] or ["lift 报告不齐全"]) + "。")
        A("")
        A("- **未放宽任何门槛**：三项判据与停机线沿用试点登记值（解析 <80% / 映射失败 >20% / 覆盖 <50% → 停），"
          "本轮**未做任何宽松匹配、未人工补录、未调分母、未剔除失败件**。")
        yrates = {y: r["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"]
                  for y in map(str, sc["ipo_years"])}
        yv = [v for v in yrates.values() if v is not None]
        ymin, ymax = (min(yv) if yv else None), (max(yv) if yv else None)
        ypass = [y for y, v in yrates.items() if v is not None and v <= 0.20]
        cats = {"碎片": fd["碎片"], "名称漂移": fd["名称漂移"], "抽取噪声": fd["抽取噪声"],
                RESIDUAL_CLS: fd.get(RESIDUAL_CLS)}
        cat_s = " / ".join(f"{k} {v}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))
        heads = {"上界（含碎片，主口径）": fd["上界（含碎片，主口径）"], "剔碎片后": fd["剔碎片后"],
                 "剔碎片与噪声后": fd["剔碎片与噪声后"], "逐年最大值": ymax}
        over = [k for k, v in heads.items() if v is not None and v > 0.20]
        under = [k for k, v in heads.items() if v is not None and v <= 0.20]
        A("- **卡点定位（描述性，不宣称单一原因；**全部按本报告数值机械读出**）**：映射失败率分列 = "
          f"上界（主口径）{_f(fd['上界（含碎片，主口径）'])}（{fd['上界分子分母']}）、剔碎片后 {_f(fd['剔碎片后'])}"
          f"（{fd['剔碎片后分子分母']}）、剔碎片与噪声后 {_f(fd['剔碎片与噪声后'])}"
          f"（{fd['剔碎片与噪声后分子分母']}）、逐年 {_f(ymin)}–{_f(ymax)}"
          f"（年内口径 ≤0.20 的年份 = {'、'.join(ypass) if ypass else '无'}）；四分类（唯一名，降序）= {cat_s}；"
          f"其中**超线列 = {'、'.join(over) if over else '无'}**、未超线列 = {'、'.join(under) if under else '无'}。"
          "细分类（抽取噪声 / 残差桶，见 §1.2）= "
          + "；".join(f"{k} {v}" for k, v in ((r.get('failure_diagnostics') or {})
                      .get("residual_split") or {}).items()) + "。"
          "**残差桶（未过前三类判据）需独立判读**：复核抽样（n=160）显示 真非上市/境外 ≈62.5% / 抽取伪影 ≈30.0% / "
          "A 股名漂移 ≈7.5%；复核量化分解 = 2027 未映射 ≈ 抽取伪影 844 / 真非 A 股 926 / A 股名漂移 257。"
          "**最宽松修正（同剔全部抽取伪影 844）后失败率 = 1183/4974 = 0.2378（复核载 0.2379）仍 > 0.20 → 停机不解除。**"
          "可能来源并列三类（不宣称单一原因）：① 口径内不可映射实体（未上市/境外/港股同业、子公司实体）；"
          "② 抽取层伪影（弱证据规则在部分年份正文/表格上误触发，通用词·科目·句片段，及桶内的句片段/公司名混合）；"
          "③ 名称漂移残余（历史简称不在代码锚定字典内）。**逐名归因需独立判读、不在本轮**；本轮已按登记执行"
          "「先修解析」两轮修复（H1/M1/M2 = registry 1.5.15/1.5.16；修 1–7 = registry 1.5.17），"
          "**修复未触及任何判据/门槛/匹配口径/分母**。")
        A("- **措辞限制保持**：**「pre-validated / 已验证」仍禁用**（外部锚条件未满足）；且与 #10 的"
          "「AI 独立判读（三代理，非人工）」证据层级限定**并列存在、互不替代**。")
    A("")
    A("## 6. 与试点（2018）的对照")
    A("")
    pc = r.get("pilot_2018_contrast")
    if pc:
        A("| 项 | 试点 2018（registry 1.5.5–1.5.7） | 全量 2018 行（本轮） |")
        A("|---|---|---|")
        rows = [("下载/定位", f"{pc['pilot']['n_downloaded']} / {pc['pilot']['n_located']}",
                 f"{pc['full_2018']['n_downloaded']} / {pc['full_2018']['n_located']}"),
                ("解析成功率", _f(pc['pilot']['parse_success_rate']), _f(pc['full_2018']['parse_success_rate'])),
                ("映射失败率", f"{_f(pc['pilot']['mapping_failure_rate'])}（{pc['pilot']['n_unmapped/n_unique']}）",
                 f"{_f(pc['full_2018']['mapping_failure_rate'])}（{pc['full_2018']['n_unmapped/n_unique']}）"),
                ("覆盖率", f"{_f(pc['pilot']['coverage_rate'])}（{pc['pilot']['n_covered/cohort']}）",
                 f"{_f(pc['full_2018']['coverage_rate'])}（{pc['full_2018']['n_covered/cohort']}）"),
                ("重合率", f"{_f(pc['pilot']['overlap_share'], '.6f')}（{pc['pilot']['pair_edge/pair_total']}）",
                 f"{_f(pc['full_2018']['overlap_share'], '.6f')}（{pc['full_2018']['pair_edge/pair_total']}）"),
                ("lift", _f(pc['pilot']['lift']), _f(pc['full_2018']['lift'])),
                ("诊断·限在网 lift", _f(pc['pilot']['diag_lift']), _f(pc['full_2018']['diag_lift'])),
                ("网络版本", pc['pilot']['network_version'], pc['full_2018']['network_version'])]
        for a, b, c in rows:
            A(f"| {a} | {b} | {c} |")
        A("")
        A(pc["note"])
        A("")
        A("- **口径一致性校验（修复轮二重跑实测，防定义漂移）**：以试点路径（`--year 2018`，同一脚本同一规则、"
          "未改任何判据）重跑 2018 → 解析 " + _f(pc['full_2018']['parse_success_rate']) + "（" +
          str(pc['full_2018']['n_located']) + "/" + str(pc['full_2018']['n_downloaded']) + "）/ 映射失败 " +
          _f(pc['full_2018']['mapping_failure_rate']) + "（" + pc['full_2018']['n_unmapped/n_unique'] + "）/ 覆盖率 " +
          _f(pc['full_2018']['coverage_rate']) + "（" + pc['full_2018']['n_covered/cohort'] + "）/ lift " +
          _f(pc['full_2018']['lift'], '.4f') + " —— 与本报告的 2018 行**逐值一致（差分 = 0）**，且与全量第一轮 ingest "
          "明细逐份对账 status 不一致 = 0 份 → 判定 **无定义漂移**；与交付版试点（99/103、0.1626、0.5905、26.4573）"
          "的差异全部来自**登记在案**的（a）解析子通道新增（OCR 回退使 2 份无文本层扫描件进入口径）、（b）修复轮二"
          "抽取层触发条件修复（R4 复合触发 / R4_tbl_head / R2 空白容错与左边界扩展）、（c）覆盖率分母口径"
          "（全量 = cohort 全量 105）。回归证据：`artifacts/tmp/tier2_full/fix2/regress_2018.json`"
          "（+ `_detail.csv`）；交付版试点产物 `reports/data_gates/G0/s3_network_prospectus_pilot_2018.*` 零改动。")
    else:
        A("（未找到试点产物，跳过对照）")
    A("")
    A("## 7. 已知限制")
    A("")
    for x in r["known_limitations"]:
        A(f"- {x}")
    A("")
    A("## 8. 复现")
    A("")
    A("```bash")
    A("python scripts/network_prospectus_anchor.py --measure --years 2015-2023 --workers 6 \\")
    A("    --ingest-detail reports/data_gates/G0/s3_network_prospectus_full_ingest_detail.csv \\")
    A("    --pre-fix-report reports/data_gates/G0/s3_network_prospectus_anchor_v1.json \\")
    A("    --pre-fix-report2 reports/data_gates/G0/s3_network_prospectus_anchor_v2_pre_fix.json")
    A("```")
    A("")
    A("前置步骤（本报告口径）：修复前报告先留档为 `s3_network_prospectus_anchor_v1.*`（标 superseded）；"
      "首次运行因名称注册表 `cache_version` 3→4 触发**强制重算**（约 2 min，缓存 `artifacts/tmp/tier2_prosp/full_name_map.json`）；"
      "locate/抽取通道（pymupdf → CID 确定性还原 → OCR 缓存）与修复前逐字一致，**不重新下载 PDF**。")
    A("")
    A(f"- 逐份明细：`{rel(path).replace('.md', '_detail.csv')}`（{sc['cohort_n']} 行：peers / mapped_peers_codes / "
      f"unmapped / channel / network_version_used）。")
    A("- 阶段报告（全量第一轮 ingest）：`reports/data_gates/G0/s3_network_prospectus_full_ingest_status.{json,md}`。")
    A("- 修复前基线（只读）：`reports/data_gates/G0/s3_network_prospectus_anchor_v1.{json,md,_detail.csv}`"
      "（superseded，修复前数值 = 本报告 `sensitivity_pre_fix`）。")
    A("")
    fx = r.get("fix_round_evidence")
    if fx:
        A("## 9. 解析层修复轮证据（H1/M1/M2；registry 1.5.15）")
        A("")
        A(fx["note"])
        A("")
        ni = fx.get("name_registry_info") or {}
        A(f"- 名称注册表（年报头部）：cache_version = **{fx.get('name_registry_cache_version')}**"
          f"（3→4 强制重算）；覆盖 {ni.get('n_codes')} 只 / {ni.get('n_files')} 份年报；"
          f"历史简称条目 {ni.get('n_short_hist')}；全称键 {ni.get('n_full_names')}；"
          f"构建耗时 {ni.get('elapsed_sec')} s；变更来源命中 = 更名披露 {ni.get('n_name_change_hits')} / "
          f"释义缩写 {ni.get('n_defn_hits')}。来源与代码锚定机制不变，**无人工补录**。")
        A(f"- M1② R2 左边界剥离（语境/连接词前缀）：命中 **{fx.get('n_r2_left_repairs')}** 条"
          "（raw → fixed，逐条留在逐份记录 `r2_left_repairs`）。")
        ex_ = fx.get("r2_left_repair_examples") or []
        if ex_:
            A("  - 高频样例：" + "、".join(f"`{x['raw→fixed']}`（{x['n']}）" for x in ex_[:10]))
        A(f"- M2 发行人自名容错剔除：**{fx.get('n_self_name_rejections')}** 条（判据 = 精确/同核/编辑距离·OCR 近似/"
          "近全包含；A 股 universe 名录守卫不做自指剔除）；形态约束/交叉校验剔除 "
          f"**{fx.get('n_shape_rejections')}** 条。")
        A("")
    rep2 = r.get("repair_round2_evidence")
    if rep2:
        A("## 10. 修复轮二证据（修 1–7；registry 1.5.17）")
        A("")
        A("修复轮二（registry 1.5.17）证据：**判据零改动**（四项定义 / 停机线（0.80 / 0.20 / 0.50 逐字不变）/ "
          "匹配口径 / 覆盖率分母 / 网络版本规则 / 边界年处理）；**不重新下载 PDF**（用既有 txt / ocr_txt / CID "
          "缓存重抽）；**无人工补录、未放宽匹配口径**。")
        A("")
        A(f"- 抽取层逐份规则计数（全量 cohort，修 1 后）：{fx.get('extraction_rule_counts') if fx else ''}")
        A(f"- R2 左边界修复（含修 1(e) 语境后缀片段/尾随「有」名录守卫剥离）命中：**{fx.get('n_r2_left_repairs') if fx else ''}** 条")
        A(f"- **修复件清单**（与上一轮逐份 diff；语料未动）：{rep2.get('n_files_changed')} 件")
        for cc, cd in (rep2.get("focus_cases") or {}).items():
            A(f"  - **{cd['year']} {cc}**（{cd.get('channel')}）：唯一名 {cd['n_unique_before_after'][0]} → "
              f"{cd['n_unique_before_after'][1]}；已映射同行 {cd['n_mapped_before_after'][0]} → {cd['n_mapped_before_after'][1]}；"
              f"新增同行名 = {'、'.join(cd['peers_added']) or '—'}；新增同行代码 = {'、'.join(cd['mapped_codes_added']) or '—'}；"
              f"新增未映射名 = {'、'.join(cd['unmapped_added'][:8]) or '—'}")
        A("")
        chg = rep2.get("changed_files") or []
        if chg:
            A("| IPO 年 | code | 通道 | 唯一名（前→后） | 已映射（前→后） | 新增同行（代码） | 新增未映射名 |")
            A("|---|---|---|---|---|---|---|")
            for cd in chg[:25]:
                A(f"| {cd['year']} | {cd['code']} | `{cd.get('channel')}` | {cd['n_unique_before_after'][0]} → "
                  f"{cd['n_unique_before_after'][1]} | {cd['n_mapped_before_after'][0]} → {cd['n_mapped_before_after'][1]} | "
                  f"{'、'.join(cd['mapped_codes_added']) or '—'} | {'、'.join(cd['unmapped_added'][:6]) or '—'} |")
            A("")
        A("- **修复轮二其余项**：修 2 残差桶改名（§1.1/§1.2/§5）；修 3 抽取噪声判据 = 整名 ≈ 通用词 + 公司后缀豁免"
          "（§1.1/§1.2）；修 4 `first_hit_page` 非 pymupdf 通道 = `not_applicable_channel`（§1 注）；修 5 报告头"
          "registry 版本 + `short_keys` 净减 1,297（报告头注）；修 6 网络节点两口径（§1 注）；修 7 "
          "`identity_check_pooled_lift` = **恒等式校验（非独立证据）**（§3 池化行）。")
        A("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
import csv
import json
import os
import re
import time

def resolve_network_version(year, net_dir):
    """逐 IPO 年 Y → 网络版本（registry 1.5.13）：y_target = min(Y+2, 2023)；
    版本文件缺失 → 以「最近可用网络版本」替代（实测 v1_2022；boundary=True 记边界年子集）。"""
    avail = sorted(int(m.group(1)) for m in
                   (re.match(r"(\d{4})_support\.csv$", x) for x in os.listdir(net_dir)) if m)
    yt = min(year + 2, 2023)
    if yt in avail:
        return yt, f"pm_ma_ewcos_teacher_v1_{yt}", False
    ynew = max(avail)
    return ynew, f"pm_ma_ewcos_teacher_v1_{ynew}", True

def _ingest_one(job):
    """逐份解析（仅解析层）：locate 三通道回退 + 逐份 channel/页数/字数/关键词记录。
    返回 dict（全量轮 ingest 明细行）；不静默剔除——失败逐条带 reason。"""
    code, year, net_ver, net_text_year, boundary, pdfd, txtd = job
    pdfp, txtp = os.path.join(pdfd, f"{code}.pdf"), os.path.join(txtd, f"{code}.txt")
    rec = {"code": code, "ipo_year": year, "network_version_used": net_ver,
           "network_text_year": net_text_year, "boundary_year_subset": boundary}
    if not (os.path.exists(pdfp) and os.path.exists(txtp)):
        rec.update(status="no_file", channel="no_file", located=False, page_count=0, char_count=0,
                   n_kw_hits=0, first_hit_page="", n_spans=0, pdf_bytes=0, txt_bytes=0,
                   reason="pdf 或 txt 缺失（下载未成功，详见 download_log.csv）")
        return rec
    t0 = time.time()
    rec.update(pdf_bytes=os.path.getsize(pdfp), txt_bytes=os.path.getsize(txtp))
    try:
        (located, n_hits, counts, first_page, n_pages, spans, text,
         offsets, channel) = locate(txtp, pdfp, code=code)
    except Exception as e:
        rec.update(status="parse_error", channel=f"error:{type(e).__name__}", located=False,
                   page_count=0, char_count=0, n_kw_hits=0, first_hit_page="", n_spans=0,
                   reason=f"locate 异常：{str(e)[:120]}")
        return rec
    insuf = _insufficient(text, n_pages)
    rec.update(status="parsed" if located else "no_section", channel=channel, located=located,
               page_count=n_pages, char_count=len(text), n_kw_hits=n_hits,
               kw_counts=json.dumps(counts, ensure_ascii=False),
               first_hit_page=first_page if first_page else "", n_spans=len(spans),
               text_insufficient=insuf, parse_sec=round(time.time() - t0, 1))
    if not located:
        if insuf:
            rec["reason"] = ("文本层缺失/不足（CID/OCR 回退后仍不足；"
                             f"company_count={text.count('公司')}, chars={len(text)}, pages={n_pages}）")
        else:
            rec["reason"] = f"章节未定位（4 关键词全 0；chars={len(text)}, pages={n_pages}）"
    return rec

def ingest_main(args):
    """全量 ingest：逐年 cohort → 并行 locate（含 OCR 回退）→ 明细 CSV + 汇总 JSON。"""
    years = []
    for part in args.years.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            years += list(range(int(a), int(b) + 1))
        elif part:
            years.append(int(part))
    years = sorted(set(years))
    firm = {}
    with open(FIRM_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            firm[r["code6"]] = r["ipo_date"]
    verb = {y: resolve_network_version(y, args.net_dir) for y in years}
    print(f"[F-3] 全量 ingest：years={years}；workers={args.workers}", flush=True)
    for y in years:
        ty, ver, bd = verb[y]
        print(f"[F-3]   {y}: cohort={sum(1 for c, d in firm.items() if d[:4] == str(y))}；"
              f"network_version_used={ver}（text_year={ty}；boundary={bd}）", flush=True)

    jobs = []
    for y in years:
        ty, ver, bd = verb[y]
        pdfd = os.path.join(args.base, str(y), "pdf")
        txtd = os.path.join(args.base, str(y), "txt")
        for c, d in sorted(firm.items()):
            if d[:4] == str(y):
                jobs.append((c, y, ver, ty, bd, pdfd, txtd))
    print(f"[F-3] 计划 {len(jobs)} 份；每 {args.progress_every} 份打印进度", flush=True)

    recs, t0 = [], time.time()
    if args.workers <= 1:
        it = ((j, _ingest_one(j)) for j in jobs)
        for i, (_, r) in enumerate(it, 1):
            recs.append(r)
            if i % args.progress_every == 0 or i == len(jobs):
                print(f"[F-3] {i}/{len(jobs)}（parsed={sum(1 for x in recs if x['status'] == 'parsed')} "
                      f"elapsed={time.time() - t0:.0f}s）", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_ingest_one, jobs, chunksize=1), 1):
                recs.append(r)
                if i % args.progress_every == 0 or i == len(jobs):
                    print(f"[F-3] {i}/{len(jobs)}（parsed="
                          f"{sum(1 for x in recs if x['status'] == 'parsed')} "
                          f"elapsed={time.time() - t0:.0f}s）", flush=True)

    outdir = args.ingest_outdir or os.path.join(ROOT, "artifacts", "tmp", "tier2_full")
    os.makedirs(outdir, exist_ok=True)
    detail_csv = args.detail_csv or os.path.join(outdir, "ingest_detail.csv")
    fields = ["code", "ipo_year", "status", "channel", "located", "page_count", "char_count",
              "n_kw_hits", "kw_counts", "first_hit_page", "n_spans", "text_insufficient",
              "pdf_bytes", "txt_bytes", "parse_sec", "network_version_used", "network_text_year",
              "boundary_year_subset", "reason"]
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x["ipo_year"], x["code"])):
            w.writerow(r)
    # 汇总
    summary = {"years": years, "n_planned": len(jobs), "workers": args.workers,
               "per_year": {}, "channel_dist": {}, "status_dist": {},
               "network_version_map": {str(y): {"text_year": verb[y][0], "version": verb[y][1],
                                                "boundary_year_subset": verb[y][2]} for y in years},
               "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_sec": round(time.time() - t0, 1)}
    ch_key = lambda c: ("pymupdf" if c == "pymupdf" else
                        "pdfminer+cid_gid→ocr" if (c.startswith("pdfminer+cid_gid")
                                                   and "ocr_tesseract_chi_sim" in c) else
                        "pdfminer+cid_gid" if c.startswith("pdfminer+cid_gid") else
                        "ocr_tesseract_chi_sim" if "ocr_tesseract_chi_sim" in c else
                        "ocr_insufficient" if "ocr_insufficient" in c else
                        "text_layer_invalid" if "text_layer_invalid" in c else
                        "no_file" if c == "no_file" else "other")
    for y in years:
        rs = [r for r in recs if r["ipo_year"] == y]
        n_dl = sum(1 for r in rs if r["status"] != "no_file")
        n_parsed = sum(1 for r in rs if r["status"] == "parsed")
        n_nofile = sum(1 for r in rs if r["status"] == "no_file")
        n_err = sum(1 for r in rs if r["status"] == "parse_error")
        ch = {}
        for r in rs:
            k = ch_key(r.get("channel", ""))
            ch[k] = ch.get(k, 0) + 1
        summary["per_year"][str(y)] = {"cohort": len(rs), "downloaded": n_dl, "parsed": n_parsed,
                                       "no_section": sum(1 for r in rs if r["status"] == "no_section"),
                                       "no_file": n_nofile, "parse_error": n_err,
                                       "parse_success_rate": round(n_parsed / n_dl, 4) if n_dl else None,
                                       "channels": ch,
                                       "boundary_year_subset": verb[y][2]}
        summary["status_dist"][str(y)] = {s: sum(1 for r in rs if r["status"] == s)
                                          for s in sorted({r["status"] for r in rs})}
        for k, v in ch.items():
            summary["channel_dist"][k] = summary["channel_dist"].get(k, 0) + v
    tot_dl = sum(v["downloaded"] for v in summary["per_year"].values())
    tot_parsed = sum(v["parsed"] for v in summary["per_year"].values())
    summary["overall"] = {"downloaded": tot_dl, "parsed": tot_parsed,
                          "parse_success_rate": round(tot_parsed / tot_dl, 4) if tot_dl else None}
    with open(os.path.join(outdir, "ingest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[F-3] 完成 {len(recs)}/{len(jobs)}：parsed={tot_parsed}/{tot_dl}"
          f"（{summary['overall']['parse_success_rate']}）channel={summary['channel_dist']} "
          f"detail={os.path.relpath(detail_csv, ROOT)}", flush=True)
    return summary
from datetime import datetime, timezone
import collections
import csv
import json
import math
import os
import time

def _init_measure_worker():
    """worker 初始化：加载 firm universe + 名称注册表（缓存）+ universe 名称键（每进程一次）。

    修 5（M2-i / registry 1.5.15）：同时构建**名称注册表按 code 倒排**（`build_self_index`），
    使发行人自名集合含「本发行人自身的代码锚定历史简称」（不再只用 universe 当前名 + 全称字典）。
    """
    firm, short = load_firm_universe()
    full, short_hist, full_info = build_full_name_map(firm)
    if not os.path.exists(CACHE) or full_info.get("cache_version") != 4:
        raise SystemExit("名称注册表缓存构建失败（cache_version != 4）")
    for nm, cs in short_hist.items():
        short.setdefault(nm, set()).update(cs)
    build_self_index(full, short_hist)
    _MEAS.update(firm=firm, short=short, full=full, uni_keys=load_universe_name_keys(),
                 full_info=full_info)

def _measure_one(job):
    """逐份测量：locate（三通道回退，与 ingest 同链）→ 抽名单 → 映射 → 分类；返回紧凑记录。"""
    code, year, ver, text_year, boundary, pdfd, txtd = job
    firm, short, full, uni_keys = _MEAS["firm"], _MEAS["short"], _MEAS["full"], _MEAS["uni_keys"]
    pdfp, txtp = os.path.join(pdfd, f"{code}.pdf"), os.path.join(txtd, f"{code}.txt")
    rec = {"code": code, "ipo_year": year, "network_version_used": ver,
           "network_text_year": text_year, "boundary_year_subset": boundary}
    empty = dict(map_modes={}, peers=[], mapped_peers_codes=[], unmapped_names=[],
                 unmapped_classified=[], rejected=[])
    if not (os.path.exists(pdfp) and os.path.exists(txtp)):
        rec.update(status="no_file", channel="no_file", located=False, page_count=0, char_count=0,
                   n_kw_hits=0, first_hit_page="", n_spans=0, n_unique_names=0, n_mapped=0,
                   n_unmapped=0, reason="pdf 或 txt 缺失（下载未成功，详见 download_log.csv）", **empty)
        return rec
    try:
        (located, n_hits, counts, first_page, n_pages, spans, text,
         offsets, channel) = locate(txtp, pdfp, code=code)
    except Exception as e:
        rec.update(status="parse_error", channel=f"error:{type(e).__name__}", located=False,
                   page_count=0, char_count=0, n_kw_hits=0, first_hit_page="", n_spans=0,
                   n_unique_names=0, n_mapped=0, n_unmapped=0,
                   reason=f"locate 异常：{str(e)[:120]}", **empty)
        return rec
    insuf = _insufficient(text, n_pages)
    rec.update(status="parsed" if located else "no_section", channel=channel, located=located,
               page_count=n_pages, char_count=len(text), n_kw_hits=n_hits, kw_counts=counts,
               first_hit_page=first_page if first_page else "", n_spans=len(spans),
               text_insufficient=insuf)
    if not located:
        rec.update(n_unique_names=0, n_mapped=0, n_unmapped=0, **empty)
        rec["reason"] = (f"文本层缺失/不足（CID/OCR 回退后仍不足；company_count={text.count('公司')}, "
                         f"chars={len(text)}, pages={n_pages}）" if insuf else
                         f"章节未定位（4 关键词全 0；chars={len(text)}, pages={n_pages}）")
        return rec
    self_exact, self_core = self_names_for(firm, code)     # 修 5（M2-i）：含代码锚定历史简称
    uni_keys = _MEAS["uni_keys"]
    REJECT_LOG.clear()
    REPAIR_LOG.clear()                                     # 修 5（M1②）：R2 左边界修复证据（逐份）
    cands = extract_candidates(text, spans, self_exact, self_core, uni_keys)
    rej = [dict(base=x["base"], peer=x.get("code"), why=x["why"]) for x in REJECT_LOG]
    REJECT_LOG.clear()
    repaired = list(REPAIR_LOG)                            # 证据：raw → fixed（不进方向性检查）
    REPAIR_LOG.clear()
    uniq = {}
    for c in cands:
        key = c["base"]
        prev = uniq.get(key)
        if prev is None or (prev["adjacent_code"] is None and c["adjacent_code"]):
            uniq[key] = c
    mapped, modes, unmapped = {}, {}, []
    for b, c in sorted(uniq.items()):
        code6, mode = map_name(c, short, full, firm, at_year=year)
        modes[mode] = modes.get(mode, 0) + 1
        if code6 == code:      # 映射到发行人自身 → 不计入自报同行（释义/正文自指，试点口径）
            continue
        if code6:
            mapped[b] = dict(code6=code6, mode=mode, raw=c["raw"])
        else:
            unmapped.append(dict(base=b, raw=c["raw"], rule=c["rule"]))
    cls_list = []
    for u in unmapped:
        cls, ev = classify_unmapped(u["base"], u["raw"], text, uni_keys, self_exact)
        cls_list.append(dict(name=u["base"], cls=cls, evidence=ev))
    rec.update(n_unique_names=len(uniq), n_mapped=len(mapped), n_unmapped=len(unmapped),
               rule_counts=dict(collections.Counter(c["rule"] for c in cands)),
               map_modes=modes, peers=sorted(mapped.keys()),
               mapped_peers_codes=sorted({v["code6"] for v in mapped.values()}),
               unmapped_names=[u["base"] for u in unmapped],
               unmapped_classified=cls_list, rejected=rej, r2_left_repairs=repaired)
    return rec

def _wilson(k, n, z=1.959963984540054):
    """Wilson score 区间（95%）；在 n = 0 时返回 (None, None)。"""
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)

def _load_network(text_year, net_dir):
    """读某网络版本的节点集/无向边集/无向对密度（与 lift_pair_density_readout 同口径）。"""
    with open(os.path.join(net_dir, f"{text_year}_support.csv"), encoding="utf-8") as f:
        sup = list(csv.DictReader(f))
    edges = set()
    with open(os.path.join(net_dir, f"{text_year}_similarity.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = r["focal_id"], r["source_id"]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    N, E = len(sup), len(edges)
    return {"text_year": text_year, "version": f"pm_ma_ewcos_teacher_v1_{text_year}",
            "N": N, "E": E, "nodes": {r["focal_id"] for r in sup}, "edges": edges,
            "d": E / (N * (N - 1) / 2) if N > 1 else float("nan")}

def _dedupe_readings(recs, label):
    """年内/池化去重口径下的名称统计：唯一名 / 未映射 / 四分类 + 分列失败率。

    修 5 ④（registry 1.5.15）：分类增第 4 类「抽取噪声」（additive、描述性）；**「剔碎片后」列口径不改**，
    另**增列「剔碎片与噪声后」**；主口径失败率（上界·含碎片）与停机线判定口径不变。
    """
    names, unmapped, cls_of = {}, {}, {}
    for r in recs:
        for nm in (r.get("peers") or []):
            names[nm] = names.get(nm, 0) + 1
        for nm in (r.get("unmapped_names") or []):
            names[nm] = names.get(nm, 0) + 1
            unmapped[nm] = unmapped.get(nm, 0) + 1
        for u in (r.get("unmapped_classified") or []):
            cls_of.setdefault(u["name"], u)
    by = {"碎片": [], "名称漂移": [], "抽取噪声": [], RESIDUAL_CLS: []}
    for n in sorted(cls_of):
        by.setdefault(cls_of[n]["cls"], []).append(cls_of[n])
    n_names, n_unm = len(names), len(unmapped)
    n_frag = len(by["碎片"])
    n_noise = len(by["抽取噪声"])
    return {
        "口径": label, "n_unique_names": n_names, "n_unmapped": n_unm,
        "上界（含碎片，主口径）": (n_unm / n_names) if n_names else None,
        "上界分子分母": f"{n_unm}/{n_names}",
        "剔碎片后": ((n_unm - n_frag) / (n_names - n_frag)) if n_names > n_frag else None,
        "剔碎片后分子分母": f"{n_unm - n_frag}/{n_names - n_frag}",
        "剔碎片与噪声后": ((n_unm - n_frag - n_noise) / (n_names - n_frag - n_noise))
                          if n_names > n_frag + n_noise else None,
        "剔碎片与噪声后分子分母": f"{n_unm - n_frag - n_noise}/{n_names - n_frag - n_noise}",
        "碎片": n_frag, "名称漂移": len(by["名称漂移"]), "抽取噪声": n_noise,
        RESIDUAL_CLS: len(by[RESIDUAL_CLS]),
        "明细": {k: [{"name": u["name"], "evidence": u["evidence"]} for u in v]
                 for k, v in by.items()},
    }

def _year_rates(rs, cohort_n=None):
    """单年四项中的前三项（解析/映射失败/覆盖）+ 通道/状态计数。"""
    n_dl = sum(1 for r in rs if r["status"] != "no_file")
    n_parsed = sum(1 for r in rs if r["status"] == "parsed")
    covered = [r for r in rs if r.get("n_mapped", 0) >= 1]
    net_nodes = [r for r in rs if r.get("network_node")]
    cov_net = [r for r in net_nodes if r.get("n_mapped", 0) >= 1]
    return {
        "cohort": len(rs), "downloaded": n_dl, "parsed": n_parsed,
        "no_section": sum(1 for r in rs if r["status"] == "no_section"),
        "no_file": sum(1 for r in rs if r["status"] == "no_file"),
        "parse_error": sum(1 for r in rs if r["status"] == "parse_error"),
        "parse_success_rate": (n_parsed / n_dl) if n_dl else None,
        "n_covered": len(covered), "coverage_rate": (len(covered) / len(rs)) if rs else None,
        "n_network_nodes": len(net_nodes), "n_covered_in_network": len(cov_net),
        # 修 6（registry 1.5.17）：网络节点**两口径**并报——located 口径 = 已定位件的在网家数；
        # 预登记会员口径 = cohort 中 firm_id ∈ 该网络版本节点集的家数（含未下载/未定位件）。
        "n_network_nodes_member_scope": sum(1 for r in rs if r.get("network_member")),
        "coverage_rate_in_network": (len(cov_net) / len(net_nodes)) if net_nodes else None,
        "peers_per_firm_mean": ((sum(r["n_mapped"] for r in rs if r.get("located"))
                                 / max(1, sum(1 for r in rs if r.get("located"))))),
        "channels": {k: sum(1 for r in rs if r.get("channel") == k)
                     for k in sorted({r.get("channel", "?") for r in rs})},
    }

def measure_main(args):
    """全量测量（汇总轮）：逐年 + 池化四项率与 lift，边界年子集单列。"""
    years = []
    for part in args.years.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            years += list(range(int(a), int(b) + 1))
        elif part:
            years.append(int(part))
    years = sorted(set(years))
    firm, short = load_firm_universe()
    full, short_hist, full_info = build_full_name_map(firm)
    CACHE_INFO.update(full_info or {})
    for nm, cs in short_hist.items():
        short.setdefault(nm, set()).update(cs)
    print(f"[M-1] 名称字典：firm universe {len(firm)}；简称键 {len(short)}；全称键 {len(full)}"
          f"（覆盖 {full_info.get('n_codes')} 只；cache_version={full_info.get('cache_version')}）", flush=True)

    verb = {y: resolve_network_version(y, args.net_dir) for y in years}
    nets = {}
    for ty in sorted({verb[y][0] for y in years}):
        nets[ty] = _load_network(ty, args.net_dir)
        print(f"[M-1] 网络 {ty}：N={nets[ty]['N']} E={nets[ty]['E']} d={nets[ty]['d']:.6f}", flush=True)

    jobs = []
    for y in years:
        ty, ver, bd = verb[y]
        pdfd = os.path.join(args.base, str(y), "pdf")
        txtd = os.path.join(args.base, str(y), "txt")
        for c, m in sorted(firm.items()):
            if m["ipo_date"][:4] == str(y):
                jobs.append((c, y, ver, ty, bd, pdfd, txtd))
    print(f"[M-1] 计划 {len(jobs)} 份（{years[0]}–{years[-1]}）；workers={args.workers}", flush=True)

    recs, t0 = [], time.time()
    if args.workers <= 1:
        _init_measure_worker()
        for i, j in enumerate(jobs, 1):
            recs.append(_measure_one(j))
            if i % args.progress_every == 0 or i == len(jobs):
                print(f"[M-2] {i}/{len(jobs)}（parsed="
                      f"{sum(1 for x in recs if x['status'] == 'parsed')} elapsed={time.time() - t0:.0f}s）",
                      flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, initializer=_init_measure_worker) as pool:
            for i, r in enumerate(pool.imap_unordered(_measure_one, jobs, chunksize=1), 1):
                recs.append(r)
                if i % args.progress_every == 0 or i == len(jobs):
                    print(f"[M-2] {i}/{len(jobs)}（parsed="
                          f"{sum(1 for x in recs if x['status'] == 'parsed')} elapsed={time.time() - t0:.0f}s）",
                          flush=True)
    recs.sort(key=lambda x: (x["ipo_year"], x["code"]))
    print(f"[M-2] 完成 {len(recs)} 份（{time.time() - t0:.0f}s）", flush=True)

    # ---- 逐份：网络对计数（focal 在网才有对；边/节点均为 firm_id 口径）----
    for r in recs:
        net = nets[verb[r["ipo_year"]][0]]
        fid = firm[r["code"]]["firm_id"]
        r["network_member"] = fid in net["nodes"]     # 修 6：预登记会员口径（不以 located 为条件）
        if not r.get("located") or fid not in net["nodes"]:
            r.update(network_node=False, n_pairs=0, n_pairs_edge=0, overlap_share=None)
            continue
        pset = sorted({firm[j]["firm_id"] for j in r.get("mapped_peers_codes") or []} - {fid})
        tot = len(pset)
        hit = sum(1 for j in pset if (min(fid, j), max(fid, j)) in net["edges"])
        pset_in = {j for j in pset if j in net["nodes"]}
        hit_in = sum(1 for j in pset_in if (min(fid, j), max(fid, j)) in net["edges"])
        r.update(network_node=True, n_pairs=tot, n_pairs_edge=hit,
                 overlap_share=(hit / tot) if tot else None,
                 n_pairs_in_network=len(pset_in), n_pairs_edge_in_network=hit_in)

    cohort_all = len(recs)
    overall = _year_rates(recs, cohort_all)
    overall["cohort"] = cohort_all
    overall["coverage_rate"] = sum(1 for r in recs if r.get("n_mapped", 0) >= 1) / cohort_all

    per_year, per_year_lift = {}, {}
    for y in years:
        rs = [r for r in recs if r["ipo_year"] == y]
        ty, ver, bd = verb[y]
        net = nets[ty]
        yr = _year_rates(rs, len(rs))
        yr["network_version_used"] = ver
        yr["network_text_year"] = ty
        yr["boundary_year_subset"] = bd
        yr["network"] = {"N": net["N"], "E": net["E"], "d": net["d"]}
        pt = sum(r.get("n_pairs", 0) for r in rs)
        ph = sum(r.get("n_pairs_edge", 0) for r in rs)
        pt_in = sum(r.get("n_pairs_in_network", 0) for r in rs)
        ph_in = sum(r.get("n_pairs_edge_in_network", 0) for r in rs)
        share = (ph / pt) if pt else None
        share_in = (ph_in / pt_in) if pt_in else None
        lo, hi = _wilson(ph, pt)
        yr.update({
            "pair_total": pt, "pair_edge": ph, "overlap_share": share,
            "lift": (share / net["d"]) if (share is not None and net["d"] == net["d"]) else None,
            "overlap_share_ci95": [lo, hi],
            "lift_ci95": [(lo / net["d"]) if lo is not None else None,
                          (hi / net["d"]) if hi is not None else None],
            "diag_pair_total": pt_in, "diag_pair_edge": ph_in, "diag_overlap_share": share_in,
            "diag_lift": (share_in / net["d"]) if (share_in is not None and net["d"] == net["d"]) else None,
        })
        per_year[str(y)] = {k: v for k, v in yr.items() if k not in ("channels",)}
        per_year_lift[str(y)] = yr
        per_year[str(y)]["channel_dist"] = yr["channels"]

    # ---- 池化：主读数年（非边界）与全期（含边界，作对照）----
    def _pooled(sel_years):
        pt = sum(per_year_lift[str(y)]["pair_total"] for y in sel_years)
        ph = sum(per_year_lift[str(y)]["pair_edge"] for y in sel_years)
        pt_in = sum(per_year_lift[str(y)]["diag_pair_total"] for y in sel_years)
        ph_in = sum(per_year_lift[str(y)]["diag_pair_edge"] for y in sel_years)
        exp = sum(per_year_lift[str(y)]["pair_total"] * per_year_lift[str(y)]["network"]["d"]
                  for y in sel_years)
        exp_in = sum(per_year_lift[str(y)]["diag_pair_total"] * per_year_lift[str(y)]["network"]["d"]
                     for y in sel_years)
        share = (ph / pt) if pt else None
        share_in = (ph_in / pt_in) if pt_in else None
        lo, hi = _wilson(ph, pt)
        d_w = (exp / pt) if pt else None
        return {
            "years": sel_years, "pair_total": pt, "pair_edge": ph,
            "overlap_share": share, "pooled_density_dw": d_w,
            "lift": (ph / exp) if exp else None,
            "overlap_share_ci95": [lo, hi],
            "lift_ci95": [(lo / d_w) if (lo is not None and d_w) else None,
                          (hi / d_w) if (hi is not None and d_w) else None],
            "diag_pair_total": pt_in, "diag_pair_edge": ph_in, "diag_overlap_share": share_in,
            "diag_lift": (ph_in / exp_in) if exp_in else None,
            # 修 7（registry 1.5.17）：原名 `weighted_mean_of_year_lifts` 与池化 lift **代数恒等**
            #（Σ(pt_y·d_y·lift_y)/Σ(pt_y·d_y) ≡ Σph/Σexp ≡ 池化 lift）→ 改名 + 标注**非独立证据**（数值保留）。
            "identity_check_pooled_lift": ((sum(per_year_lift[str(y)]["pair_total"]
                                                * per_year_lift[str(y)]["network"]["d"]
                                                * (per_year_lift[str(y)]["lift"] or 0)
                                                for y in sel_years) / exp) if exp else None),
            "identity_check_pooled_lift_note": "恒等式校验（非独立证据）：Σ(pt_y·d_y·lift_y)/Σ(pt_y·d_y) "
                                               "≡ Σph/Σexp ≡ 池化 lift；仅作代数一致性校验，不得当独立读数引用。",
        }

    main_years = [y for y in years if not verb[y][2]]
    boundary_years = [y for y in years if verb[y][2]]
    pooled_main = _pooled(main_years)
    pooled_all = _pooled([y for y in years if per_year_lift[str(y)]["pair_total"]])
    boundary = {}
    for y in boundary_years:
        py = per_year_lift[str(y)]
        boundary[str(y)] = {
            "cohort": py["cohort"], "downloaded": py["downloaded"], "parsed": py["parsed"],
            "n_network_nodes": py["n_network_nodes"], "pair_total": py["pair_total"],
            "overlap_share": py["overlap_share"], "lift": py["lift"],
            "network_version_used": py["network_version_used"],
            "note": "边界年（y_target = min(Y+2, 2023) 不可得 → fallback v1_2022）：cohort 在该网络版本中"
                    "节点数 = 0（冻结样本筛选 ipo<2y）→ 锚读数退化（无对）；只报下载/解析/覆盖等计数，"
                    "不参与池化与主读数。",
        }

    # ---- 映射失败率分列（池化全局 / 主读数年 / 边界年 / 逐年）----
    fail_all = _dedupe_readings(recs, "全量池化（2015–2023 跨年去重，主口径）")
    fail_main = _dedupe_readings([r for r in recs if not verb[r["ipo_year"]][2]],
                                 "主读数年池化（2015–2020 跨年去重，诊断）")
    fail_bnd = _dedupe_readings([r for r in recs if verb[r["ipo_year"]][2]],
                                "边界年子集（2021–2023 跨年去重，单列）")
    fail_year = {str(y): _dedupe_readings([r for r in recs if r["ipo_year"] == y],
                                          f"{y} 年内去重") for y in years}

    # ---- 通道对照（与 ingest 明细逐份核对；不一致即口径漂移证据）----
    cross = {"ingest_detail": "", "n_compared": 0, "status_mismatch": [], "channel_mismatch": []}
    if args.ingest_detail and os.path.exists(args.ingest_detail):
        cross["ingest_detail"] = rel(args.ingest_detail)
        ing = {r["code"]: r for r in csv.DictReader(open(args.ingest_detail, encoding="utf-8"))}
        for r in recs:
            g = ing.get(r["code"])
            if not g:
                cross["status_mismatch"].append({"code": r["code"], "ingest": "missing"})
                cross["n_compared"] += 1
                continue
            cross["n_compared"] += 1
            if g["status"] != r["status"]:
                cross["status_mismatch"].append({"code": r["code"], "ingest": g["status"],
                                                 "measure": r["status"]})
            gi = g["channel"]
            if gi != r["channel"]:
                cross["channel_mismatch"].append({"code": r["code"], "ingest": g["channel"],
                                                  "measure": r["channel"]})

    # ---- 下载失败概览（19 件；逐条 reason 来自各年 download_log.csv）----
    dl = {"n_failed": 0, "by_status": {}, "items": []}
    for y in years:
        lp = os.path.join(args.base, str(y), "download_log.csv")
        if not os.path.exists(lp):
            continue
        last = {}
        for row in csv.DictReader(open(lp, encoding="utf-8")):
            last[row.get("code", "")] = row
        for r in recs:
            if r["ipo_year"] != y or r["status"] != "no_file":
                continue
            row = last.get(r["code"], {})
            st = row.get("status", "unknown")
            dl["by_status"][st] = dl["by_status"].get(st, 0) + 1
            dl["items"].append({"year": y, "code": r["code"], "status": st,
                                "tried": (row.get("tried") or "")[:120]})
            dl["n_failed"] += 1

    # ---- 非 pymupdf 通道逐件披露（OCR / CID）----
    chan = [{"year": r["ipo_year"], "code": r["code"], "channel": r["channel"],
             "pages": r["page_count"], "chars": r["char_count"], "kw_hits": r["n_kw_hits"],
             "status": r["status"]} for r in recs if r.get("channel") not in ("pymupdf", "no_file")]

    # ---- 方向性检查（全量）：被剔名-码对（双方均在网）落网率 vs 保留集 ----
    rej = {}
    for r in recs:
        for x in (r.get("rejected") or []):
            p = x.get("peer")
            if p and p in firm and r["code"] in firm:
                fid = firm[r["code"]]["firm_id"]
                j = firm[p]["firm_id"]
                ty = verb[r["ipo_year"]][0]
                if fid != j and fid in nets[ty]["nodes"] and j in nets[ty]["nodes"]:
                    rej.setdefault((ty, fid, j), x["base"])
    rej_hit = sum(1 for (ty, i, j) in rej if (min(i, j), max(i, j)) in nets[ty]["edges"])

    # ---- 停机线与外部锚条件（机械判定，不得放宽）----
    th = {"parse_min": 0.80, "map_fail_max": 0.20, "coverage_min": 0.50}
    stop = []
    if overall["parse_success_rate"] < th["parse_min"]:
        stop.append(f"解析成功率 {overall['parse_success_rate']:.4f} < {th['parse_min']}")
    if fail_all["上界（含碎片，主口径）"] > th["map_fail_max"]:
        stop.append(f"映射失败率 {fail_all['上界（含碎片，主口径）']:.4f} > {th['map_fail_max']}")
    if overall["coverage_rate"] < th["coverage_min"]:
        stop.append(f"覆盖率 {overall['coverage_rate']:.4f} < {th['coverage_min']}")
    lift_complete = bool(pooled_main.get("lift") is not None
                         and all(per_year[str(y)]["lift"] is not None for y in main_years)
                         and boundary)
    external = {
        "criteria": {
            "解析成功率 ≥0.80": {"value": overall["parse_success_rate"],
                                 "pass": overall["parse_success_rate"] >= th["parse_min"]},
            "映射失败率 ≤0.20": {"value": fail_all["上界（含碎片，主口径）"], "scope": "全量池化上界口径",
                                 "pass": fail_all["上界（含碎片，主口径）"] <= th["map_fail_max"]},
            "覆盖率 ≥0.50": {"value": overall["coverage_rate"], "scope": "分母 = 2015–2023 cohort 2,656",
                              "pass": overall["coverage_rate"] >= th["coverage_min"]},
        },
        "lift_reports_complete": lift_complete,
        "stop_triggered": bool(stop), "stop_reasons": stop,
        "external_anchor_satisfied": (not stop) and lift_complete,
    }

    # ---- 失败构成诊断（描述性，不改任何主口径）----
    ucnt, ucls, uyr = collections.Counter(), {}, {}
    for r in recs:
        for nm in (r.get("unmapped_names") or []):
            ucnt[nm] += 1
            uyr.setdefault(nm, set()).add(r["ipo_year"])
        for u in (r.get("unmapped_classified") or []):
            ucls.setdefault(u["name"], u)
    split = {"疑似抽取噪声（整名 ≈ 通用词/科目/句片段词）": 0, RESIDUAL_CLS: 0}
    for nm in ucnt:
        c = (ucls.get(nm) or {}).get("cls")
        if c == "抽取噪声":
            split["疑似抽取噪声（整名 ≈ 通用词/科目/句片段词）"] += 1
        elif c == RESIDUAL_CLS:
            split[RESIDUAL_CLS] += 1
    n_resid = split[RESIDUAL_CLS]
    n_noise_u = split["疑似抽取噪声（整名 ≈ 通用词/科目/句片段词）"]
    failure_diag = {
        "note": "描述性诊断（不改主口径、不参与任何率/门槛判定）：① unmapped 按文档频次排序（识别高频噪声 token）；"
                "② 分类第 4 类「抽取噪声」（registry 1.5.17 修 3：判据 = **整名 ≈ 通用词**（整名 ∈ 词表 或 ≤"
                + str(GEN_SUB_MAXLEN) + " 字且含词表词）+ **公司后缀豁免**（公司/集团/股份/有限/企业 收尾或含 股份/集团））单列；"
                "③ **残差桶（未过前三类判据）**（registry 1.5.17 修 2 改名）——**兜底 return**，其来源**需独立判读**，"
                "**不得**以本桶断言「非 A 股实体」：复核抽样审计（n=160，从该桶随机抽）显示 真非上市/境外 ≈62.5%、"
                "抽取伪影 ≈30.0%、A 股名漂移 ≈7.5%；复核量化分解（外推）见 residual_bucket_audit。",
        "top_unmapped_by_docfreq": [{"name": nm, "n_docs": c, "cls": (ucls.get(nm) or {}).get("cls"),
                                     "years": sorted(uyr.get(nm, [])),
                                     "evidence": (ucls.get(nm) or {}).get("evidence")}
                                    for nm, c in ucnt.most_common(40)],
        "residual_split": split,
        "rename_note": "原字段 `non_a_share_split` / 原类名「非 A 股实体」= 兜底断言；registry 1.5.17 修 2 改为中性命名"
                       "「残差桶（未过前三类判据）」，字段名同步为 `residual_split`（数值口径未改，仅命名与文案）。",
        "residual_bucket_audit": {
            "bucket_n": n_resid,
            "sampling_audit": {"n_sampled": 160, "真非上市/境外": 100, "抽取伪影": 48, "A股名漂移": 12,
                               "share": {"真非上市/境外": 0.625, "抽取伪影": 0.300, "A股名漂移": 0.075},
                               "source": "独立复核（2026-10-14 第二轮）随机抽样 160 名人工/AI 独立判读"},
            "quantified_decomposition": {
                "note": "复核量化分解（**基于上一轮/修复前四分类（碎片 282 / 名称漂移 146 / 抽取噪声 117 / "
                        "残差桶（旧名「非 A 股实体」）1482）+ n=160 抽样外推**，口径 = 修复前 2027 未映射 / 5818 唯一名）："
                        "2027 未映射唯一名 ≈ 抽取伪影 844（= 桶内伪影外推 445（30.0%×1482）+ 上一轮抽取噪声 117 + "
                        "上一轮碎片 282）/ 真非 A 股 926（= 桶内 62.5%×1482）/ A 股名漂移 257（= 桶内漂移外推 111（7.5%×1482）+ "
                        "上一轮名称漂移 146）。**本轮修 3 重分类后四分类计数另计**（碎片 "
                        + str(fail_all["碎片"] if isinstance(fail_all, dict) else "—") + " / 名称漂移 "
                        + str(fail_all["名称漂移"] if isinstance(fail_all, dict) else "—") + " / 抽取噪声 " + str(n_noise_u)
                        + " / 残差桶 " + str(n_resid) + "），故最宽松修正读数沿用**复核口径**（= 修复前分母 5818）并注记。",
                "抽取伪影": 844, "真非A股": 926, "A股名漂移": 257, "合计": 2027,
                "loosest_correction": {"rate": (2027 - 844) / (5818 - 844), "numerator_denominator": "1183/4974",
                                       "review_value": 0.2379,
                                       "note": "最宽松修正（同剔全部抽取伪影 844）后仍 >0.20 → **停机不解除**"},
            },
        },
        "n_unmapped_unique": len(ucnt),
        "outcome_reading": (
            f"本轮实测（唯一名口径，修复后）：抽取噪声 {n_noise_u} 个（判据 = 整名 ≈ 通用词 + 公司后缀豁免）/ "
            f"残差桶（未过前三类判据）{n_resid} 个；该桶**需独立判读**——复核抽样（n=160）显示其中 "
            "真非上市/境外 ≈62.5%、抽取伪影 ≈30.0%、A 股名漂移 ≈7.5%，即**不是单一来源**；"
            "最宽松修正（同剔全部抽取伪影 844）后失败率 = 1183/4974 ≈ 0.2378（复核载 0.2379）**仍 > 0.20** → "
            "**停机不解除**、外部锚条件仍不满足（机械判定，未放宽任何门槛）。"),
    }

    # ---- 修复轮证据（H1/M1/M2；描述性，不改任何口径）----
    n_r2_rep = sum(len(r.get("r2_left_repairs") or []) for r in recs)
    r2_ex = collections.Counter()
    for r in recs:
        for x in (r.get("r2_left_repairs") or []):
            r2_ex[f"{x['raw']}→{x['fixed']}"] += 1
    n_self_rej = sum(1 for r in recs for x in (r.get("rejected") or [])
                     if str(x.get("why", "")).startswith("发行人自指"))
    rule_tot = collections.Counter()
    for r in recs:
        for k, v in (r.get("rule_counts") or {}).items():
            rule_tot[k] += v
    fix_evidence = {
        "note": "registry 1.5.15「先修解析」修复轮证据（解析层规则化修复；**无人工补录、未放宽匹配口径、未剔失败件**）："
                "H1 = 年报头部名称注册表切分（cache_version 3→4 强制重算）；M1 = 招股书抽取规则（R2 标签集 + 左边界剥离 + "
                "R5 整名噪声判据）；M2 = 发行人自名过滤容错（含代码锚定历史简称）。"
                "**修复轮二（registry 1.5.17）**：修 1 抽取层触发条件（R4 复合触发 + 和/与/及 并列切分 + R2 括号内空白容错 + "
                "R2 左边界语境后缀片段/尾随「有」（名录守卫）剥离 + 新增 R4_tbl_head）——逐份规则计数见 `extraction_rule_counts`。",
        "extraction_rule_counts": dict(sorted(rule_tot.items())),
        "name_registry_cache_version": CACHE_INFO.get("cache_version"),
        "name_registry_info": CACHE_INFO,
        "n_r2_left_repairs": n_r2_rep,
        "r2_left_repair_examples": [{"raw→fixed": k, "n": v} for k, v in r2_ex.most_common(15)],
        "n_self_name_rejections": n_self_rej,
        "n_shape_rejections": sum(1 for r in recs for x in (r.get("rejected") or [])
                                  if not str(x.get("why", "")).startswith("发行人自指")),
    }

    # ---- 与试点 2018 的对照 ----
    pilot_cmp = None
    ppath = os.path.join(ROOT, "reports", "data_gates", "G0", "s3_network_prospectus_pilot_2018.json")
    if os.path.exists(ppath):
        pj = json.load(open(ppath, encoding="utf-8"))
        py = per_year_lift["2018"]
        pfd = (pj["rates"].get("failure_decomposition") or {})
        pilot_cmp = {
            "pilot_artifact": rel(ppath),
            "pilot": {"n_downloaded": pj["rates"]["n_downloaded"], "n_located": pj["rates"]["n_located"],
                      "parse_success_rate": pj["rates"]["parse_success_rate"],
                      "mapping_failure_rate": pj["rates"]["mapping_failure_rate"],
                      "n_unmapped/n_unique": f"{pj['rates']['n_unmapped_names']}/{pj['rates']['n_unique_names']}",
                      "coverage_rate": pj["rates"]["coverage_rate"],
                      "n_covered/cohort": f"{pj['rates']['n_covered']}/{pj['rates']['cohort_n']}",
                      "overlap_share": pj["anchor_readout"]["overlap_share_raw"],
                      "pair_edge/pair_total": f"{pj['anchor_readout']['pair_edge']}/{pj['anchor_readout']['pair_total']}",
                      "lift": pj["anchor_readout"]["lift"],
                      "diag_lift": pj["anchor_readout"]["diagnostic_node_restricted"]["lift"],
                      "network_version": pj["scope"]["network_version"]},
            "full_2018": {"n_downloaded": py["downloaded"], "n_located": py["parsed"],
                          "parse_success_rate": py["parse_success_rate"],
                          "coverage_rate": py["coverage_rate"],
                          "n_covered/cohort": f"{py['n_covered']}/{py['cohort']}",
                          "mapping_failure_rate": fail_year["2018"]["上界（含碎片，主口径）"],
                          "n_unmapped/n_unique": fail_year["2018"]["上界分子分母"],
                          "overlap_share": py["overlap_share"],
                          "pair_edge/pair_total": f"{py['pair_edge']}/{py['pair_total']}",
                          "lift": py["lift"], "diag_lift": py["diag_lift"],
                          "network_version": py["network_version_used"]},
            "note": "试点口径（103 份下载 / 99 定位 / 重合率 0.555985 → lift 26.4573）与全量 2018 行的差异"
                    "来源 = ① 全量轮补 OCR 通道使 2 份无文本层扫描件（300760/300748）进入口径（cohort 105 全量），"
                    "② 抽取/映射规则与通道逐字一致、网络版本同为 v1_2020 → 差异只来自新增 2 份件与"
                    "「全量为 cohort 全量、试点为下载成功口径」的分母差。",
        }

    low_cov = [y for y in years if (per_year[str(y)]["coverage_rate"] or 0) < 0.50]
    known_limitations = [
        "**本轮为登记在案的「先修解析」修复轮**（registry 1.5.15；H1/M1/M2 三处解析缺陷的规则化修复）："
        "**判据零改动**（四项定义 / 停机线 / 匹配口径 / 覆盖率分母 / 网络版本规则 / 边界年处理一律不改）；"
        "修复前基线逐值见 `sensitivity_pre_fix`（含修复前后对照，§0）；修复**不解除任何措辞限制**——"
        "外部锚条件按修复后读数机械重出、不加宽任何门槛。",
        "**覆盖率为下界**：覆盖率分母（2015–2023 cohort 2,656 ∩ 项目 universe）固定，分子只随抽取/映射成功"
        f"而增加；未定位（{overall['no_section']} 份 no_section）、下载未成功（{overall['no_file']} 份 no_file）"
        "与抽到名单但无可映射同行只会让覆盖率更低。",
        f"**逐年覆盖率提示（口径并报）**：全量覆盖率（分母 = 2,656 池化）= {overall['coverage_rate']:.4f} 过线；"
        f"但按 workplan 规划期措辞「**逐年**名单覆盖率 <50% → 停」的字面口径，"
        + ("、".join(f"{y} 年 = {per_year[str(y)]['coverage_rate']:.4f}（{per_year[str(y)]['n_covered']}/"
                    f"{per_year[str(y)]['cohort']}）" for y in low_cov) if low_cov else "无年份低于 0.50")
        + "。registry 冻结的 `stop_rules` 为池化口径「覆盖率 <50%」（判据真源），两类口径均在本报告并报；"
          "无论按哪种口径，**映射失败率一项已独立超线** → 外部锚条件未满足的判定不变。",
        "**lift 方向未证**（沿用试点结论）：漏抽同行同时减少分子（漏掉的同行对）与分母（自报同行对总数），"
        "净方向取决于漏抽同行的落网率相对全体自报同行落网率的高低；全量方向性检查见 anchor_readout.direction_check，"
        f"被剔名-码对落网率 = {(rej_hit / len(rej)) if rej else None}（保留集诊断口径 = "
        f"{pooled_main.get('diag_overlap_share')}）→ 不宣称下界、不宣称上界。",
        "**CI 为指示性**：Wilson 区间按对（pair）二项抽样计算，未校正「同一 focal 内多对相关」的聚类效应；"
        "池化 lift 的 CI 只传递重合率的抽样不确定性，基线密度 d 视为固定。",
        "**SW2 快照与本步无关**：本锚只用「招股书自报名单 ↔ 主口径边表」的对照，不涉及申万二级行业快照"
        "（same_sw2 等诊断指标），故「SW2 快照版本」问题不影响本报告任何读数。",
        f"**OCR 回退件 {len([x for x in chan if 'ocr' in x['channel']])} 件 / CID 确定性还原件 "
        f"{len([x for x in chan if x['channel'].startswith('pdfminer+cid_gid')])} 件**（共 {len(chan)} 件非 pymupdf 通道）："
        "逐件通道/页数/字数/关键词命中见 §通道披露；OCR 为 tesseract 5.3.4 chi_sim（--psm 3、200 DPI），"
        "**非人工、非宽松匹配**；OCR 文本可能含识别噪声（已知：公司与代码识别错误率高于文本层）。",
        f"**下载失败 {dl['n_failed']} 件**（status 概览 = {dl['by_status']}）：逐条 reason 见 §下载失败概览"
        "与 `data/raw/cninfo_prospectus/<year>/download_log.csv`；**未静默剔除**（全部计入 cohort 与覆盖率分母）。",
        "**边界年（2021–2023）锚读数退化**：y_target = min(Y+2, 2023) 中 v1_2023 不存在 → fallback v1_2022；"
        "三 cohort 在该网络版本中节点数 = 0（冻结样本筛选「上市不足 2 年」）→ 无对、lift 不可测；"
        "边界年只报下载/解析/覆盖计数，单列不混入主读数。",
        "**修复轮二（registry 1.5.17，修 1–7）**：**判据零改动**（四项定义 / 停机线 / 匹配口径 / 覆盖率分母 / 网络版本规则 / "
        "边界年处理一律不改）；只修**抽取层触发条件**与**归因判据/文案**，**无人工补录、未放宽匹配、未调分母、未剔失败件**；"
        "修复前基线（上一轮 H1/M1/M2 后的读数）逐值见 `sensitivity_pre_round2`（更早的 v1 基线见 `sensitivity_pre_fix`）；"
        "修复件清单与新增同行对见 `repair_round2_evidence`。",
        "**修 1 修复件（曾漏抽、已修）**：① **300748**（上市 2018，OCR 件）：原抽 0 名（正文「同行业可比公司中科三环、宁波韵升、"
        "正海磁材、英洛华、京磁股份和大地能」无「包括/有：」触发词、末位成员含「和」→ 现由 R4 复合触发 + 和字切分 + R4_tbl_head 抽出）；"
        "② **300737**（上市 2018，CID 件）：原 4 名同行仅抽 1（东方雨虹 002271 因「（证\n\n券代码：002271）」断行 + 左边界语境「上市公司有」"
        "→ 现由 R2 括号内空白容错 + 左边界扩展抽出，凯伦股份 300715 保持）。两件前后对照逐值见 `repair_round2_evidence.focus_cases`。",
        "**残差桶（未过前三类判据）需独立判读（修 2）**：该桶是 `classify_unmapped` 的**兜底 return**，不是判据证据；"
        "复核抽样（n=160）显示 真非上市/境外 ≈62.5% / 抽取伪影 ≈30.0% / A 股名漂移 ≈7.5% → **不得**以该桶断言失败主体为单一来源；"
        "复核量化分解（2027 未映射 ≈ 抽取伪影 844 / 真非 A 股 926 / A 股漂移 257）与**最宽松修正后 0.2378（复核载 0.2379）仍 >0.20**"
        "见 `failure_diagnostics.residual_bucket_audit`。",
        "**抽取噪声判据（修 3）**：判据 = 「整名 ≈ 通用词」（整名 ∈ 词表 或 ≤5 字且含词表词）+ **公司后缀豁免**"
        "（公司/集团/股份/有限/企业 收尾或含 股份/集团）——原「子串命中」判据把 117 条中的 73 条（≥6 字或带 公司/集团/股份，"
        "如 上海弘陆物流设备 / 山东联创节能新材料）误判为抽取噪声，现归入残差桶（**该分列仍为描述性规则判读口径**，主口径失败率不依赖它）。",
        "**字段语义（修 4）**：`first_hit_page` 对**非 pymupdf 通道**（CID 确定性还原 / OCR 回退）置 `not_applicable_channel`"
        "（替换后文本的字符偏移与 pymupdf 页码不对应，不得以页码冒充）；pymupdf 通道仍报真实页码。",
        "**网络节点两口径（修 6）**：`n_network_nodes`（**located 口径**）只数已定位件中的在网家数；"
        "`n_network_nodes_member_scope`（**预登记会员口径**）= cohort 中 firm_id ∈ 该网络版本节点集的家数（含未下载/未定位件）——"
        "两口径并报、不得混用（预登记轮登记值 = 1,537 / 1,592 = 96.5%）。",
        "**恒等式校验（修 7）**：`identity_check_pooled_lift` 与池化 lift **代数恒等**（非独立证据；原名 "
        "`weighted_mean_of_year_lifts`），只作一致性校验，不得当独立读数引用。",
        "**修 1 新增抽取的粘连片段（如实披露，未做事后剔除）**：修复轮二新增抽取的未映射名中约半数属「真名 + 粘连前缀/后缀」"
        "（如 莱克电气显 / 选择兆龙互连 / 定期报告 / 公开资料），与既有已知残余「太原重式截断名」同类（R4 列举型 token 为"
        "无代码锚定的字面切片，遇换行/长词即截断）；该项使**主口径失败率上升**（上一轮 0.3484 → 本轮 0.3670，方向上不利于过线），"
        "且已计入分母与分子——**未按任何判据事后剔除、未人工补录**。",
        "**已知残余**：① 「太原重」式截断名（形态上与真名不可分）；② 表格列错行的码-名错位（名与码都在同行集内但配对错位）；"
        "③ 抽取口径偏保守（未含未上市/境外/港股同业）→ 覆盖率与重合率为下界口径。",
        "**角色限定**：本锚只作外部一致性读数，不进入面板/回归、不改主口径；"
        "「边级语义」表述仍受 #10 证据层级（AI 独立判读，三代理、非人工）限定约束。",
    ]

    # ---- 修复前基线（registry 1.5.15 修复轮 F3：前置/后对照；只读引用，不改本轮任何读数）----
    pre = None
    ppre = args.pre_fix_report
    if ppre and os.path.exists(ppre):
        try:
            p = json.load(open(ppre, encoding="utf-8"))
            pfd = p["rates"]["failure_decomposition_pooled"]
            plp = p["anchor_readout"]["pooled_main_years_2015_2020"]
            pre = {
                "source": rel(ppre), "run_at_before_fix": p["run_at"],
                "note": "修复前（registry 1.5.13/1.5.14 全量第二轮）读数；口径与判据逐字未改，"
                        "差异全部来自解析层修复（H1/M1/M2，registry 1.5.15 登记）。"
                        "修复前分类为**三分类**（碎片/名称漂移/非 A 股实体）——本轮新增第 4 类「抽取噪声」"
                        "（作用位置 = 原「非 A 股实体」桶内细分，additive），故修复前「非 A 股实体」"
                        "1,584 个内含两类（形态为公司名 1,470 / 通用词·科目·句片段 114）。",
                "parse_success_rate": p["rates"]["overall"]["parse_success_rate"],
                "parse_n": f"{p['rates']['overall']['parsed']}/{p['rates']['overall']['downloaded']}",
                "mapping_failure_rate_upper": pfd["上界（含碎片，主口径）"],
                "mapping_failure_n": pfd["上界分子分母"],
                "mapping_failure_ex_frag": pfd["剔碎片后"],
                "mapping_failure_ex_frag_n": pfd["剔碎片后分子分母"],
                "cls3": {"碎片": pfd["碎片"], "名称漂移": pfd["名称漂移"], "非 A 股实体": pfd["非 A 股实体"]},
                "coverage_rate": p["rates"]["overall"]["coverage_rate"],
                "coverage_n": f"{p['rates']['overall']['n_covered']}/{p['rates']['overall']['cohort']}",
                "pooled_lift": plp["lift"], "pooled_pair": f"{plp['pair_edge']}/{plp['pair_total']}",
                "pooled_overlap_share": plp["overlap_share"],
                "external_anchor_satisfied": p["external_anchor"]["external_anchor_satisfied"],
                "per_year": {y: {"parse_success_rate": p["rates"]["per_year"][y]["parse_success_rate"],
                                 "coverage_rate": p["rates"]["per_year"][y]["coverage_rate"],
                                 "mapping_failure_rate": p["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"],
                                 "lift": p["anchor_readout"]["per_year"][y]["lift"]}
                             for y in map(str, years)},
            }
        except Exception as e:      # 只读对照，失败不阻断本轮（报告注明缺失）
            pre = {"source": rel(ppre), "error": f"{type(e).__name__}: {str(e)[:120]}"}

    # ---- 修复轮二：上一轮（H1/M1/M2）基线 + 修复件清单/新增同行对（只读对照，不改本轮读数）----
    pre2, repair2 = None, None
    if args.pre_fix_report2 and os.path.exists(args.pre_fix_report2):
        try:
            q = json.load(open(args.pre_fix_report2, encoding="utf-8"))
            qfd = q["rates"]["failure_decomposition_pooled"]
            qlp = q["anchor_readout"]["pooled_main_years_2015_2020"]
            qres = [k for k in qfd.keys() if k.startswith("残差桶") or k == "非 A 股实体"]
            qres_key = qres[0] if qres else "非 A 股实体"
            pre2 = {
                "source": rel(args.pre_fix_report2), "run_at_before_fix2": q["run_at"],
                "note": "上一轮（H1/M1/M2 解析层修复轮，registry 1.5.15/1.5.16）修复后读数 = **本轮修复前基线**；"
                        "本轮（修复轮二，registry 1.5.17）差异全部来自修 1（抽取层触发条件）/ 修 2（命名）/"
                        "修 3（抽取噪声判据）——判据（四项定义/停机线/匹配口径/分母/网络版本规则）逐字未改。",
                "parse_success_rate": q["rates"]["overall"]["parse_success_rate"],
                "parse_n": f"{q['rates']['overall']['parsed']}/{q['rates']['overall']['downloaded']}",
                "mapping_failure_rate_upper": qfd["上界（含碎片，主口径）"],
                "mapping_failure_n": qfd["上界分子分母"],
                "mapping_failure_ex_frag": qfd["剔碎片后"],
                "mapping_failure_ex_frag_n": qfd["剔碎片后分子分母"],
                "mapping_failure_ex_frag_noise": qfd.get("剔碎片与噪声后"),
                "mapping_failure_ex_frag_noise_n": qfd.get("剔碎片与噪声后分子分母"),
                "cls4": {"碎片": qfd["碎片"], "名称漂移": qfd["名称漂移"], "抽取噪声": qfd["抽取噪声"],
                         qres_key: qfd[qres_key]},
                "coverage_rate": q["rates"]["overall"]["coverage_rate"],
                "coverage_n": f"{q['rates']['overall']['n_covered']}/{q['rates']['overall']['cohort']}",
                "pooled_lift": qlp["lift"], "pooled_pair": f"{qlp['pair_edge']}/{qlp['pair_total']}",
                "pooled_overlap_share": qlp["overlap_share"],
                "external_anchor_satisfied": q["external_anchor"]["external_anchor_satisfied"],
                "per_year": {y: {"parse_success_rate": q["rates"]["per_year"][y]["parse_success_rate"],
                                 "coverage_rate": q["rates"]["per_year"][y]["coverage_rate"],
                                 "mapping_failure_rate": q["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"],
                                 "lift": q["anchor_readout"]["per_year"][y]["lift"]}
                             for y in map(str, years)},
            }
            # 修复件清单与新增同行对（修 1 定点证据；逐件 diff，只读）
            qf = {x["code"]: x for x in (q.get("per_firm") or [])}
            changed = []
            for r in recs:
                o = qf.get(r["code"])
                if not o:
                    continue
                if not r.get("located"):
                    continue
                op = set(o.get("peers") or [])
                np_ = set(r.get("peers") or [])
                ou = set(o.get("unmapped_names") or [])
                nu = set(r.get("unmapped_names") or [])
                ocode = set(o.get("mapped_peers_codes") or [])
                ncode = set(r.get("mapped_peers_codes") or [])
                if op != np_ or ou != nu or ocode != ncode:
                    changed.append({
                        "year": r["ipo_year"], "code": r["code"], "channel": r.get("channel"),
                        "n_unique_before_after": [o.get("n_unique_names"), r.get("n_unique_names")],
                        "n_mapped_before_after": [o.get("n_mapped"), r.get("n_mapped")],
                        "peers_added": sorted(np_ - op), "peers_removed": sorted(op - np_),
                        "mapped_codes_added": sorted(ncode - ocode), "mapped_codes_removed": sorted(ocode - ncode),
                        "unmapped_added": sorted(nu - ou)[:20], "unmapped_removed": sorted(ou - nu)[:20],
                        "r2_left_repairs": [f"{x['raw']}→{x['fixed']}" for x in (r.get("r2_left_repairs") or [])][:10],
                    })
            changed.sort(key=lambda x: (-len(x["peers_added"]), x["code"]))
            focus = {c["code"]: c for c in changed if c["code"] in ("300748", "300737")}
            repair2 = {
                "note": "修复轮二（修 1）修复件清单与新增同行对：与上一轮（registry 1.5.16）逐份 diff（只读）；"
                        "**语料未动（不重新下载 PDF）**，差异全部来自抽取层触发条件修复。",
                "n_files_changed": len(changed),
                "focus_cases": focus,
                "changed_files": changed[:60],
                "diff_source": rel(args.pre_fix_report2),
            }
        except Exception as e:      # 只读对照，失败不阻断本轮（报告注明缺失）
            pre2 = {"source": rel(args.pre_fix_report2), "error": f"{type(e).__name__}: {str(e)[:160]}"}

    result = {
        "report": "第二梯队 #9 路线 B 全量（2015–2023）招股书「同行业可比公司」自报锚（锚 1 / plan §8 B″11-7）",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "parameter": "prospectus_peer_anchor_pilot（registry 1.5.17 全量口径 + 修复轮二（修 1–7）；"
                     "其中 H1/M1/M2 解析层修复轮 = registry 1.5.15/1.5.16；四项定义 / 停机线 / 匹配口径逐字沿用"
                     "试点登记值，**判据零改动**）",
        "fix_round": {
            "registry_version": "1.5.15",
            "change": "第二梯队 #9 全量「先修解析」修复轮：H1（年报头部名称注册表「简称+全称同头拼接」错切 + 标签边界切分 + "
                      "cache_version 3→4）/ M1（R2 标签集补证券代码·公司代码 + R2 左边界剥离 + R5 单字噪声判据 → 整名判据）/ "
                      "M2（发行人自名过滤容错：自名集合含代码锚定历史简称 + 同核/编辑距离·OCR 近似/近全包含 + universe 名录守卫）",
            "judgment_unchanged": "四项定义 / 停机线（0.80 / 0.20 / 0.50 逐字不变）/ 匹配口径（精确 + 仅公司后缀归一化 + "
                                  "代码锚定，无模糊匹配）/ 覆盖率分母（cohort 2,656）/ 网络版本规则（min(Y+2,2023) + fallback v1_2022）/ "
                                  "边界年处理；**无人工补录、未放宽匹配、未调分母、未剔除失败件**",
            "attribution_additive": "归因口径增设第 4 类「抽取噪声」（判据 = GEN_SUB 词表；作用位置 = 原「非 A 股实体」桶内细分提为顶层类），"
                                    "另增列「剔碎片与噪声后」；主口径失败率（上界·含碎片）与停机线判定口径不变",
        },
        "fix_round2": {
            "registry_version": "1.5.17",
            "change": "第二梯队 #9 全量第二轮修复轮（修 1–7）：① 修 1 抽取层触发条件（R4 补「可比公司」类复合触发 + "
                      "和/与/及 并列切分 + R2 括号内空白容错 + R2 左边界语境后缀片段/尾随「有」（名录守卫）剥离 + "
                      "新增「可比公司表头列」型 R4_tbl_head；R1 触发条件复核后不改）；② 修 2 unmapped 第 4 类改名"
                      "「残差桶（未过前三类判据）」+ 抽样审计（n=160）与复核量化分解入报告；③ 修 3 GEN_SUB 判据改"
                      "「整名 ≈ 通用词」+ 公司后缀豁免；④ 修 4 first_hit_page 非 pymupdf 通道置 not_applicable_channel；"
                      "⑤ 修 5 报告头 registry 版本改实际登记版本 + short_keys 净减 1,297 登记；⑥ 修 6 网络节点两口径"
                      "并报；⑦ 修 7 恒等式校验改名（非独立证据）",
            "judgment_unchanged": "四项定义 / 停机线（0.80 / 0.20 / 0.50 逐字不变）/ 匹配口径（精确 + 仅公司后缀归一化 + "
                                  "代码锚定，无模糊匹配）/ 覆盖率分母（cohort 2,656）/ 网络版本规则（min(Y+2,2023) + "
                                  "fallback v1_2022）/ 边界年处理；**无人工补录、未放宽匹配、未调分母、未剔除失败件**",
            "re_extract_scope": "用既有 txt / ocr_txt / CID 缓存**重抽**（不重新下载 PDF）；修复件清单与新增同行对见 "
                                "`repair_round2_evidence`",
        },
        "sensitivity_pre_fix": pre,
        "sensitivity_pre_round2": pre2,
        "repair_round2_evidence": repair2,
        "scope": {
            "ipo_years": years, "cohort_n": cohort_all, "sample_scope": "全量（不抽样）",
            "universe": "data/processed/firm_id_map.csv（项目 firm universe）",
            "download_dir": rel(os.path.join(args.base)),
            "network_version_map": {str(y): {"text_year": verb[y][0], "version": verb[y][1],
                                             "boundary_year_subset": verb[y][2]} for y in years},
            "keyword_set": list(KWS), "window": [WIN_PRE, WIN_POST],
            "extraction_rules": ["R1_full", "R2_name_code", "R3_code_name", "R4_enum", "R4_tbl_head",
                                 "R5_table_row"],
            "r1_trigger_recheck": "修 1(g)：R1（全称型）触发词与近邻窗口本轮**已复核、无改动**"
                                  "（复核点名两件的漏抽由 R4 复合触发 / R4_tbl_head / R2 空白容错 / R2 左边界扩展覆盖）",
            "mapping_modes": ["exact_short", "exact_full", "suffix_norm", "adjacent_code"],
            "name_dict": {"firm_universe": len(firm), "short_keys": len(short), "full_keys": len(full)},
            "name_dict_short_keys_note": "修 5：`name_dict.short_keys` 净减 1,297（H1 错切键移除，13,412→12,115；"
                                         "键数下降来自错切键与头部标签残片键移除，**非**人工删除）",
            "channels": "pymupdf → pdfminer+SimSun GID（CID 编码损坏件，确定性还原）→ OCR 回退"
                        "（tesseract 5.3.4 chi_sim，--psm 3、200 DPI）；不足判据 = 公司<20 或 字符数<max(5000,50×页数)",
            "parse_layer_fix": "registry 1.5.15「先修解析」修复轮（H1 名称注册表切分 + M1 抽取规则 + M2 自名容错）；"
                               "locate/通道/匹配口径/判据与修复前逐字一致",
        },
        "rates": {"overall": overall, "thresholds": th, "stop_triggered": bool(stop),
                  "stop_reasons": stop,
                  "failure_decomposition_pooled": fail_all,
                  "failure_decomposition_main_years": fail_main,
                  "failure_decomposition_boundary": fail_bnd,
                  "failure_decomposition_per_year": fail_year,
                  "per_year": per_year},
        "anchor_readout": {
            "definition": "lift = 重合率 ÷ 该网络年无偏对密度 d = E/(N(N−1)/2)；池化 lift = Σ命中对 ÷ Σ(该年对数 × d_年) "
                          "（= 密度加权平均 lift，等价于池化重合率 ÷ 池化密度）",
            "per_year": {str(y): {k: per_year_lift[str(y)][k] for k in
                                  ("network_version_used", "network_text_year", "pair_total", "pair_edge",
                                   "overlap_share", "overlap_share_ci95", "lift", "lift_ci95",
                                   "diag_pair_total", "diag_pair_edge", "diag_overlap_share", "diag_lift",
                                   "n_network_nodes", "network")} for y in years},
            "pooled_main_years_2015_2020": pooled_main,
            "pooled_including_boundary": pooled_all,
            "boundary_subset": boundary,
            "per_firm_overlap_stats": {
                "n_firms_in_network_with_peers": sum(1 for r in recs if r.get("network_node")
                                                    and r.get("overlap_share") is not None),
                "n_zero_overlap": sum(1 for r in recs if r.get("network_node")
                                      and r.get("overlap_share") == 0),
                "n_ge_50pct": sum(1 for r in recs if r.get("network_node")
                                  and (r.get("overlap_share") or 0) >= 0.5),
                "mean": (sum(r["overlap_share"] for r in recs if r.get("overlap_share") is not None)
                         / max(1, sum(1 for r in recs if r.get("overlap_share") is not None))),
            },
            "direction_check": {
                "note": "方向性检查（全量）：被 R3/R2 形态约束或 R1/R2 交叉校验剔除的名-码对（双方均在该网络年节点集内）"
                        "落网率 vs 保留集诊断口径 — 若二者同量级，说明剔除会改变（而非仅压低）重合率 → lift 方向未证。",
                "rejected_pairs_in_network": len(rej), "rejected_pairs_edge": rej_hit,
                "rejected_pairs_edge_rate": (rej_hit / len(rej)) if rej else None,
                "kept_pairs_edge_rate_diag": pooled_main.get("diag_overlap_share"),
                "rejected_examples": sorted({f"{k[0]}:{k[1]}↔{k[2]}（{v}）" for k, v in rej.items()})[:15],
            },
        },
        "external_anchor": external,
        "failure_diagnostics": failure_diag,
        "fix_round_evidence": fix_evidence,
        "ingest_crosscheck": cross,
        "pilot_2018_contrast": pilot_cmp,
        "download_failures": dl,
        "channel_disclosure": chan,
        "known_limitations": known_limitations,
        "per_firm": recs,
    }

    out = args.out or os.path.join(ROOT, "reports", "data_gates", "G0",
                                   "s3_network_prospectus_anchor.json")
    out_md = args.out_md or out.replace(".json", ".md")
    detail_csv = args.detail_csv or out.replace(".json", "_detail.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def _atomic_write(path, writer):
        """原子写（先写 .tmp 再 os.replace）：防并发/中断造成的半写文件。"""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    _atomic_write(out, lambda f: json.dump(result, f, ensure_ascii=False, indent=1))
    json.load(open(out, encoding="utf-8"))      # 写后回读校验（防半写/编码损坏）
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "ipo_year", "status", "channel", "located", "page_count", "char_count",
                    "n_kw_hits", "first_hit_page", "n_unique_names", "n_mapped", "n_unmapped",
                    "network_version_used", "network_text_year", "boundary_year_subset",
                    "network_node", "n_pairs", "n_pairs_edge", "overlap_share",
                    "peers", "mapped_peers_codes", "unmapped"])
        for r in recs:
            w.writerow([r["code"], r["ipo_year"], r["status"], r.get("channel", ""), r.get("located", ""),
                        r.get("page_count", ""), r.get("char_count", ""), r.get("n_kw_hits", ""),
                        r.get("first_hit_page", ""), r.get("n_unique_names", ""), r.get("n_mapped", ""),
                        r.get("n_unmapped", ""), r["network_version_used"], r["network_text_year"],
                        r["boundary_year_subset"], r.get("network_node", ""), r.get("n_pairs", ""),
                        r.get("n_pairs_edge", ""), r.get("overlap_share", ""),
                        "|".join(r.get("peers") or []), "|".join(r.get("mapped_peers_codes") or []),
                        "|".join(r.get("unmapped_names") or [])])
    write_anchor_md(result, out_md)
    with open(out.replace(".json", "_measure.log"), "w", encoding="utf-8") as f:
        f.write(json.dumps({k: result[k] for k in ("run_at", "rates", "anchor_readout",
                                                   "external_anchor", "scope")},
                           ensure_ascii=False, indent=1))
    print(f"[M-3] 报告 → {rel(out)} / {rel(out_md)}；逐份明细 → {rel(detail_csv)}", flush=True)
    print(f"[M-3] 解析 {overall['parse_success_rate']:.4f}（{overall['parsed']}/{overall['downloaded']}）"
          f"/ 映射失败 {fail_all['上界（含碎片，主口径）']:.4f}（{fail_all['上界分子分母']}）"
          f"/ 覆盖 {overall['coverage_rate']:.4f}（{overall['n_covered']}/{cohort_all}）", flush=True)
    print(f"[M-3] 池化 lift（主读数年 {main_years}）= {pooled_main['lift']}；external_anchor_satisfied = "
          f"{external['external_anchor_satisfied']}", flush=True)
    return result


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---
import argparse
import collections
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone

import pymupdf



# 修 5（registry 1.5.15）：R2 标签集补「证券代码 / 公司代码」（原仅「股票代码」）——
# 招股书正文常见「名称（证券代码：NNNNNN）」形态（如 300737 正文「东方雨虹（证券代码：002271）…」）。
# 修 1（2026-10-14 修复轮二，registry 1.5.17）：R2 括号内**空白容错**——代码标签内 /
# 标签与数字之间允许空白与换行断行（CID/OCR 件常见「（证\n\n券代码：002271）」；
# 300737 因该断行整条不命中 → 东方雨虹 002271 漏抽）。标签集与匹配形态不变。
# 修 1（registry 1.5.17）(a)：R4 触发词补「可比公司」类**复合触发词**——招股书常见
# 「同行业可比公司A、B、C、D、E和F」直列（无「包括/有：」），原触发词集不命中（300748 原抽 0 名）。
# 修 1（registry 1.5.17）(b)：并列切分补「和/与/及」（末位成员原整 token 含连接词被 `_raw_ok` 丢弃；
# 切分后逐 token 仍走同一形态过滤 → 含连接词的 token 照样被丢，**不放宽匹配口径**）。
# 修 1（registry 1.5.17）(f)：「可比公司表头列」型（`R4_tbl_head`）——表头行首 token = 可比公司类词，
# 行内其余 token 均为期间/数字类；取其后**连续** ≥2 行（行首 2–10 汉字短名 + 行内含数字/百分号）的行首名。
# 修 1(a) 守则（registry 1.5.17）：复合触发词直列时（无可信语境前置），成员须形如**公司简称**——
# 散文续写（「同行业可比公司不存在明显差异，主要由于…」「可比公司上市后发展较快，盈利能力、…」）不得入名单。
# 虚词字符（成员全名含任一即判散文片段）：真实公司简称不含 的/了/较/均/在/于/为/是… 等虚词。
# 注：**不含** 有/会/再/同/大/多/通/达/实/包/中/上/高 等（真实简称可作首字/中字：有研新材 / 会稽山 /
# 再升科技 / 同花顺 / 大族激光 / 多氟多 / 中科三环 …）——只留纯虚词，避免误伤真名。






# 抽取层精度过滤：财务科目/表格通用词的**子串**黑名单（命中即弃；公司名不含这些子串）
# 弱证据规则（R4 列举 / R5 表格行，无代码锚定）的加严词表：科目/资产/费用/客户分部件类
# 连接词/句式子串：出现即说明该 token 是句子片段而非公司名（公司名不含这些连接成分）


# ---- 修 5（2026-10-14 修复轮，registry 1.5.15「先登记后跑」）：H1/M1/M2 三处解析缺陷的规则化修复 ----
# H1-(b) 标签边界切分：年报头部副标题行无空白，「简称」捕获串常与紧随的代码类标签粘连
#（如「公司代码：603501公司简称：韦尔股份转债代码：113616转债简称：韦尔转债…」）→ 按最早标签边界切分。
#（与 §P5 PLACE_PREFIX 同源；PLACE_PREFIX 收录 2 字形态（黑龙/内蒙），此处按登记原文补 3 字前缀
#（黑龙江/内蒙古）——登记原文：「区域/『中国』前缀（与 §P5 PLACE_PREFIX 同源，**含黑龙江/内蒙古等 3 字前缀**）」）
# 修 5（registry 1.5.15）④：归因口径第 4 类「抽取噪声」判据 = 既有通用词/科目/句片段词表
#（与第一轮 `failure_diagnostics.GEN_SUB` 同表；此处提为模块级单一真源）。
# 修 3（registry 1.5.17）：GEN_SUB 判据 = **整名 ≈ 通用词**（+ 公司后缀豁免）。




# 修 2（registry 1.5.17）：unmapped 兜底类**改名**——原「非 A 股实体」是断言而非判据证据。





















# ---- 修 1（2026-10-13 修正轮）：R3「代码+短名」名称形态约束 ----
# 背景：R3 只要求「6 位代码后随 2–10 字中文」，会把**表格/正文片段**当成短名（复核实测 14 例，
# 如 601606 的「属于铁路」「钢丝」「太原重」、300724 的「半导体集成电路制造设」、300741 的「食用香精」）。
# 约束 = ① 长度 3–10 汉字；② 不得为动词/连接词/主语词起头；③ 不含业务/产品/工序描述词（「主营业务」列片段）；
# ④ 与 R1/R2 交叉校验（不得是本文档 R1/R2 候选名的真片段）。













# ---------------------------------------------------------------- dictionaries






# ---------------------------------------------------------------- P2 locate






# ------------------------------------------------- OCR 回退通道（全量轮新增；registry 1.5.13 预登记）
# 引擎实测选型 = WSL Ubuntu tesseract 5.3.4（chi_sim, --psm 3；渲染 200 DPI；经 wsl.exe 调用）。
# 仅当文本层缺失/不足（`公司`<20 或 字符数 < max(5000, 50×页数)）且 CID 确定性还原不可用时启用。
                           # 曾用 4 → 24 并发 tesseract × 默认 OMP 线程 → load 95/32、0.12 页/s，
                           # 见 ocr_tesseract_chunk.sh 的 OMP_THREAD_LIMIT=2）










# ---------------------------------------------------------------- P3 extract




# ---------------------------------------------------------------- P4 map




# ---------------------------------------------------------------- P5 classify
# 修 2（2026-10-13 修正轮）：unmapped 名三分类 = 抽取碎片 / 名称漂移 / 非 A 股实体。
# 碎片判据（规则 + 复核清单）：① 正文中总以「前接汉字」出现（截断残片）；② 起头 公司/本公司/发行人；
# ③ 含体裁词（招股书/说明书）；④ 含科目/句片段词（信息来源/销售数量/竞争力…）。
# 名称尾（公司简称常见收尾词）——用于「与 A 股 universe 名同头 2 字 + 公司式收尾」的改名判读
# 地名/区域前缀（2 字判据排除项：地名同头不是改名证据）与非实体后缀词（子公司/设施/项目实体）








# ---------------------------------------------------------------- 全量轮 ingest（registry 1.5.13）






# ---------------------------------------------------------------- 全量测量（registry 1.5.13 汇总轮）
# 口径（逐字沿用试点，判据零改动）：四项定义 / 停机线 / 匹配口径（精确 + 后缀归一化 + 代码锚定）、
# 覆盖率分母 = 2015–2023 cohort 全量（2,656）；逐 IPO 年 Y 用其 network_version_used（min(Y+2,2023)
# + fallback v1_2022）；边界年子集 {2021,2022,2023} 单列、不混入主读数。




















def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--codes", default="")
    ap.add_argument("--base", default=os.path.join(ROOT, "data", "raw", "cninfo_prospectus"))
    ap.add_argument("--net-dir", default=NET_DIR)
    ap.add_argument("--text-lag", type=int, default=2, help="网络文本年 = 试点年 + lag（冻结样本筛选口径）")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--detail-csv", default="")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--years", default="", help="全量 ingest 模式（registry 1.5.13）：如 2015-2023；"
                                                "只出解析层记录，不做锚测量（测量留汇总轮）")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（--years / --measure 模式）")
    ap.add_argument("--ingest-outdir", default="", help="ingest 输出目录（缺省 artifacts/tmp/tier2_full；"
                                                        "分批跑时每年批写独立目录，便于合并/断点续跑）")
    ap.add_argument("--measure", action="store_true",
                    help="全量测量模式（汇总轮）：抽取+映射+四项率+逐年/池化 lift+边界年单列+最终锚报告")
    ap.add_argument("--ingest-detail", default=os.path.join(ROOT, "reports", "data_gates", "G0",
                                                            "s3_network_prospectus_full_ingest_detail.csv"),
                    help="阶段报告逐份明细（--measure 模式下作 channel/status 逐份对照）")
    ap.add_argument("--pre-fix-report", default=os.path.join(ROOT, "reports", "data_gates", "G0",
                                                             "s3_network_prospectus_anchor_v1.json"),
                    help="修复前锚报告（只读；--measure 模式下嵌入 sensitivity_pre_fix 前置/后对照）")
    ap.add_argument("--pre-fix-report2", default=os.path.join(ROOT, "reports", "data_gates", "G0",
                                                              "s3_network_prospectus_anchor_v2_pre_fix.json"),
                    help="上一轮（H1/M1/M2 修复轮）锚报告（只读；修复轮二的前置基线 sensitivity_pre_round2）")
    args = ap.parse_args()

    if args.measure:
        return measure_main(args)
    if args.years:
        return ingest_main(args)

    year = args.year
    text_year = year + args.text_lag
    net_ver = f"pm_ma_ewcos_teacher_v1_{text_year}"
    datadir = os.path.join(args.base, str(year))
    pdfd, txtd = os.path.join(datadir, "pdf"), os.path.join(datadir, "txt")
    out = args.out or os.path.join(ROOT, "reports", "data_gates", "G0",
                                   f"s3_network_prospectus_pilot_{year}.json")
    out_md = args.out_md or out.replace(".json", ".md")
    detail_csv = args.detail_csv or os.path.join(ROOT, "artifacts", "tmp", "tier2_prosp",
                                                 f"pilot_{year}_detail.csv")

    firm, short = load_firm_universe()
    full, short_hist, full_info = build_full_name_map(firm)
    if not os.path.exists(CACHE) or full_info.get("cache_version") != 4:
        raise SystemExit("名称注册表缓存构建失败（cache_version != 4）")
    for nm, cs in short_hist.items():
        short.setdefault(nm, set()).update(cs)
    uni_keys = load_universe_name_keys()
    build_self_index(full, short_hist)     # 修 5（M2-i）：自名集合含代码锚定历史简称（试点路径同链）
    print(f"[P-2] 名称字典：firm universe {len(firm)}；简称键 {len(short)}（含历史简称 {len(short_hist)}）；"
          f"全称键 {len(full)}（{full_info.get('n_codes')} 只覆盖）", flush=True)

    cohort = [c for c, m in firm.items() if m["ipo_date"][:4] == str(year)]
    cohort.sort()
    if args.codes:
        keep = set(args.codes.split(","))
        cohort = [c for c in cohort if c in keep]
    print(f"[P-2] 试点年 {year} IPO 公司（项目 universe 内）= {len(cohort)} 家；网络版本 = {net_ver}", flush=True)

    # 网络
    with open(os.path.join(args.net_dir, f"{text_year}_support.csv"), encoding="utf-8") as f:
        sup = list(csv.DictReader(f))
    with open(os.path.join(args.net_dir, f"{text_year}_similarity.csv"), encoding="utf-8") as f:
        edges = set()
        for r in csv.DictReader(f):
            a, b = r["focal_id"], r["source_id"]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    N = len(sup)
    E = len(edges)
    base_density = E / (N * (N - 1) / 2) if N > 1 else float("nan")
    nodes = {r["focal_id"] for r in sup}
    print(f"[P-5] 网络 {text_year}：N={N} 无向边 E={E} 基线密度 d={base_density:.6f}", flush=True)

    detail, t0 = [], time.time()
    n_downloaded = 0
    n_located = 0
    all_names, all_unmapped = {}, {}
    pair_tot, pair_hit, pair_tot_in, pair_hit_in = 0, 0, 0, 0
    rejected = []
    per_firm = []
    for i, code in enumerate(cohort):
        pdfp, txtp = os.path.join(pdfd, f"{code}.pdf"), os.path.join(txtd, f"{code}.txt")
        rec = dict(code=code, firm_id=firm[code]["firm_id"], name=firm[code]["name"],
                   ipo_date=firm[code]["ipo_date"], board=firm[code]["board"])
        if not (os.path.exists(pdfp) and os.path.exists(txtp)):
            rec.update(status="no_file", located=False, n_unique_names=0, n_mapped=0, n_unmapped=0,
                       network_node=firm[code]["firm_id"] in nodes, n_pairs=0, n_pairs_edge=0,
                       overlap_share=None, unmapped_names=[], unmapped_classified=[], channel="no_file")
            detail.append(rec)
            per_firm.append(rec)
            continue
        n_downloaded += 1
        (located, n_hits, counts, first_page, n_pages, spans, text,
         offsets, channel) = locate(txtp, pdfp, code=code)
        rec.update(status="parsed" if located else "no_section", located=located, n_kw_hits=n_hits,
                   kw_counts=counts, first_hit_page=first_page, n_pages=n_pages,
                   n_spans=len(spans), text_chars=len(text), channel=channel)
        if not located:
            rec.update(n_unique_names=0, n_mapped=0, n_unmapped=0,
                       network_node=firm[code]["firm_id"] in nodes, n_pairs=0, n_pairs_edge=0,
                       overlap_share=None, unmapped_names=[], unmapped_classified=[])
            detail.append(rec)
            per_firm.append(rec)
            if (i + 1) % args.progress_every == 0 or i + 1 == len(cohort):
                print(f"[P-2] 进度 {i + 1}/{len(cohort)}（located={n_located}）", flush=True)
            continue
        n_located += 1
        self_exact, self_core = self_names_for(firm, code)   # 修 5（M2-i）：含代码锚定历史简称
        REJECT_LOG.clear()
        REPAIR_LOG.clear()
        cands = extract_candidates(text, spans, self_exact, self_core, uni_keys)
        for x in REJECT_LOG:
            rejected.append(dict(focal=code, base=x["base"], peer=x.get("code"), why=x["why"]))
        REJECT_LOG.clear()
        REPAIR_LOG.clear()
        uniq = {}
        for c in cands:
            key = c["base"]
            prev = uniq.get(key)
            if prev is None or (prev["adjacent_code"] is None and c["adjacent_code"]):
                uniq[key] = c
        mapped, modes, unmapped = {}, {}, []
        for b, c in sorted(uniq.items()):
            code6, mode = map_name(c, short, full, firm, at_year=year)
            modes[mode] = modes.get(mode, 0) + 1
            if code6 == code:      # 映射到发行人自身 → 不计入自报同行（释义/正文自指）
                continue
            if code6:
                mapped[b] = dict(code6=code6, mode=mode, raw=c["raw"])
            else:
                unmapped.append(dict(base=b, raw=c["raw"], rule=c["rule"]))
            all_names[b] = all_names.get(b, 0) + 1
            if not code6:
                all_unmapped[b] = all_unmapped.get(b, 0) + 1
        rec.update(n_unique_names=len(uniq), n_mapped=len(mapped), n_unmapped=len(unmapped),
                   map_modes=modes, peers=sorted(mapped.keys()),
                   peers_codes=sorted({v["code6"] for v in mapped.values()}),
                   unmapped_names=[u["base"] for u in unmapped][:20],
                   unmapped_classified=[dict(name=u["base"], cls=classify_unmapped(
                       u["base"], u["raw"], text, uni_keys, self_exact)[0],
                       evidence=classify_unmapped(u["base"], u["raw"], text, uni_keys, self_exact)[1])
                       for u in unmapped])
        # P5 lift 分量（节点与边均为 firm_id 口径）
        fid = firm[code]["firm_id"]
        if fid in nodes:
            pset = sorted({firm[j]["firm_id"] for j in {v["code6"] for v in mapped.values()}} - {fid})
            tot = len(pset)
            hit = sum(1 for j in pset if (min(fid, j), max(fid, j)) in edges)
            rec.update(network_node=True, n_pairs=tot, n_pairs_edge=hit,
                       overlap_share=(hit / tot) if tot else None)
            pair_tot += tot
            pair_hit += hit
            pset_in = {j for j in pset if j in nodes}
            pair_tot_in += len(pset_in)
            pair_hit_in += sum(1 for j in pset_in if (min(fid, j), max(fid, j)) in edges)
        else:
            rec.update(network_node=False, n_pairs=0, n_pairs_edge=0, overlap_share=None)
        per_firm.append(rec)
        detail.append(rec)
        if (i + 1) % args.progress_every == 0 or i + 1 == len(cohort):
            print(f"[P-2] 进度 {i + 1}/{len(cohort)}（located={n_located} "
                  f"names={len(all_names)} unmapped={len(all_unmapped)} elapsed={time.time() - t0:.0f}s）",
                  flush=True)

    # ---------------- 三项率 ----------------
    parse_rate = n_located / n_downloaded if n_downloaded else float("nan")
    n_names = len(all_names)
    map_fail_rate = len(all_unmapped) / n_names if n_names else float("nan")
    covered = [r for r in per_firm if r.get("n_mapped", 0) >= 1]
    coverage_rate = len(covered) / len(cohort) if cohort else float("nan")
    cov_in_net = [r for r in per_firm if r.get("network_node") and r.get("n_mapped", 0) >= 1]
    n_net_nodes = sum(1 for r in per_firm if r.get("network_node"))
    coverage_in_net = len(cov_in_net) / n_net_nodes if n_net_nodes else float("nan")
    share = pair_hit / pair_tot if pair_tot else float("nan")
    share_in = pair_hit_in / pair_tot_in if pair_tot_in else float("nan")
    lift = share / base_density if base_density and base_density == base_density else float("nan")
    lift_in = share_in / base_density if base_density and base_density == base_density else float("nan")

    stop = []
    if parse_rate < 0.80:
        stop.append(f"解析成功率 {parse_rate:.4f} < 0.80")
    if map_fail_rate > 0.20:
        stop.append(f"映射失败率 {map_fail_rate:.4f} > 0.20")
    if coverage_rate < 0.50:
        stop.append(f"覆盖率 {coverage_rate:.4f} < 0.50")

    # 失败案例样本（≤10）：解析失败（无文件/未定位）→ 抽到名单但无可映射同行 → 在网但零重合
    fail_cases = []
    for r in per_firm:
        if r.get("status") == "no_file":
            fail_cases.append(dict(case="no_file_scan_only_or_download_failed", code=r["code"],
                                   name=r["name"], detail="PDF/txt 缺失（实测为无文本层扫描件或候选下载失败）"))
        elif not r.get("located"):
            ch = str(r.get("channel", ""))
            if "备选通道失败" in ch:
                det = f"文本层 CID 编码损坏且备选通道失败（{ch}）；{r.get('n_pages')} 页"
            elif r.get("kw_counts") and not any(r["kw_counts"].values()):
                det = f"4 关键词命中数 = 全 0（{r.get('n_pages')} 页）——非标准/缺失章节标题"
            else:
                det = f"4 关键词命中数 = 全 0（{r.get('n_pages')} 页）"
            fail_cases.append(dict(case="section_not_located", code=r["code"], name=r["name"], detail=det))
    for r in per_firm:
        if r.get("located") and r.get("n_mapped", 0) == 0:
            fail_cases.append(dict(case="no_mapped_peer", code=r["code"], name=r["name"],
                                   detail="、".join(r.get("unmapped_names", [])[:5])
                                   or f"唯一名称 {r.get('n_unique_names', 0)} 个（均被自身/黑名单过滤）"))
    for r in per_firm:
        if r.get("network_node") and (r.get("overlap_share") == 0):
            fail_cases.append(dict(case="zero_overlap_in_network", code=r["code"], name=r["name"],
                                   detail="自报同行（已映射）" + "、".join(r.get("peers", [])[:6])))
    fail_cases = fail_cases[:10]

    unmapped_top = sorted(all_unmapped.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    year_counts = {}
    for r in detail:
        year_counts[r.get("status", "?")] = year_counts.get(r.get("status", "?"), 0) + 1

    # ---------------- 修 2 ②：失败率口径分列（碎片 / 名称漂移 / 非 A 股实体） ----------------
    cls_of = {}
    for r in per_firm:
        for u in (r.get("unmapped_classified") or []):
            cls_of.setdefault(u["name"], u)
    by_cls = {"碎片": [], "名称漂移": [], RESIDUAL_CLS: []}
    for n, u in sorted(cls_of.items()):
        by_cls.setdefault(u["cls"], []).append(u)
    n_frag = len(by_cls["碎片"])
    fail_no_frag = ((len(all_unmapped) - n_frag) / (n_names - n_frag)) if n_names > n_frag else float("nan")
    failure_decomposition = {
        "note": "修 2：unmapped 唯一名三分类（规则判据 + 复核清单；明细含逐名证据）。"
                "「剔碎片后」= 分子分母同剔碎片（碎片不是公司名，不应进入名称口径）",
        "上界（含碎片，主口径）": map_fail_rate, "上界分子分母": f"{len(all_unmapped)}/{n_names}",
        "剔碎片后": fail_no_frag, "剔碎片后分子分母": f"{len(all_unmapped) - n_frag}/{n_names - n_frag}",
        "碎片": len(by_cls["碎片"]), "名称漂移": len(by_cls["名称漂移"]),
        RESIDUAL_CLS: len(by_cls[RESIDUAL_CLS]),
        "明细": {k: [{"name": u["name"], "evidence": u["evidence"]} for u in v] for k, v in by_cls.items()},
        "修正前对照（v4 交付口径）": {"含碎片上界": 0.1869, "含碎片分子分母": "63/337",
                                     "剔碎片后（复核清单 12 碎片）": 51 / 325},
    }

    # 覆盖率拆解与逐公司重合率分布
    cov_breakdown = {
        "no_file（下载失败，不计入解析分母）": year_counts.get("no_file", 0),
        "no_section（解析未定位，计入解析分母）": year_counts.get("no_section", 0),
        "located_but_no_mapped_peer": sum(1 for r in per_firm if r.get("located") and r.get("n_mapped", 0) == 0),
        "covered（非空且 ≥1 已映射）": len(covered),
    }
    ov = [r["overlap_share"] for r in per_firm
          if r.get("network_node") and r.get("overlap_share") is not None]
    ov_stats = {
        "n_firms_in_network_with_peers": len(ov),
        "mean": (sum(ov) / len(ov)) if ov else None,
        "median": (sorted(ov)[len(ov) // 2] if ov else None),
        "min": min(ov) if ov else None, "max": max(ov) if ov else None,
        "n_zero_overlap": sum(1 for x in ov if x == 0),
        "n_ge_50pct": sum(1 for x in ov if x >= 0.5),
        "n_ge_90pct": sum(1 for x in ov if x >= 0.9),
    }
    peers_per_firm = [r["n_mapped"] for r in per_firm if r.get("located")]
    mode_totals = {}
    for r in per_firm:
        for k, v in (r.get("map_modes") or {}).items():
            mode_totals[k] = mode_totals.get(k, 0) + v
    # 解析修复轮次（停机 → 先修解析 → 复测；日志为证据）
    parse_iterations = [
        {"round": "v1 初始抽取规则", "mapping_failure_rate": 0.4681, "n_unmapped": 293,
         "n_unique": 626, "coverage_rate": 0.6762, "lift": 26.0008,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis.log",
         "note": "窗口 hit−150..+1500；无形态过滤（连接词/数字/科目词未过滤）；R5 表格行无同行语境要求"},
        {"round": "v2 收窄窗口 + R1 近邻 + 兄弟前缀", "mapping_failure_rate": 0.3653, "n_unmapped": 183,
         "n_unique": 501, "coverage_rate": 0.6667, "lift": 26.1124,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis_v2.log",
         "note": "窗口 hit−100..+500；R1 只取命中近邻（+250 字内）；R5 增加同行语境词与兄弟前缀过滤"},
        {"round": "v3 形态过滤生效 + 弱证据规则加严", "mapping_failure_rate": 0.2382, "n_unmapped": 81,
         "n_unique": 340, "coverage_rate": 0.5905, "lift": 25.3663,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis_v3.log",
         "note": "接上 `_raw_ok` 形态过滤（连接词/数字/括号/句片段）；R4 列举必须紧跟同行语境词；R5 语境 60 字 + "
                 "科目/资产/费用词表加严"},
        {"round": "v4 名称注册表补历史简称（代码锚定）", "mapping_failure_rate": 0.1869,
         "n_unmapped": 63, "n_unique": 337, "coverage_rate": 0.5905, "lift": 25.7681,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis_v4.log",
         "note": "年报头部「股票简称+股票代码」配对抽取历史简称（代码锚定、非模糊匹配），弥合 2018 名称与"
                 "2026 项目名单的名称漂移（v4 为**交付口径**，其后经独立复核 CONFIRMED_WITH_ISSUES）"},
        {"round": "v5 修正轮（本轮）：R3 名称形态约束 + 名称注册表标签/顺序/更名/释义 + CID 备选通道",
         "mapping_failure_rate": map_fail_rate, "n_unmapped": len(all_unmapped), "n_unique": n_names,
         "coverage_rate": coverage_rate, "lift": lift,
         "log": "artifacts/tmp/tier2_prosp_fix/f1_dump_v2.log / f1_variant_eval3.py / f3_recovery.log",
         "note": "① 修 1：R3「代码+短名」加名称形态约束（3–10 汉字、非动词/连接词/科目词起头、无业务描述词）"
                 "并与 R1/R2 交叉校验（真片段弃），剔除复核确认的抽取污染；② 修 2：名称注册表头部窗口 6000→24000 字、"
                 "标签补 公司代码/A股证券简称 且允许冒号、头部「简称+全称同头拼接」去重、新增更名披露与释义缩写两条"
                 "代码锚定来源、注册名去 ST/*ST 前缀建键；③ 修 3：3 份文本层 CID 编码损坏件走 pdfminer+GID 反查"
                 "备选通道（解析成功率上升）"},
    ]

    # ---------------- 修 1：抽取污染登记与敏感性（修正前/复核/修正后三套读数） ----------------
    r3_pollution = {
        "issue": "R3「6 位代码 + 2–10 字中文」把表格/正文片段误当短名（独立复核 CONFIRMED_WITH_ISSUES 实测 14 例，"
                 "经相邻代码误挂为同行）",
        "examples": ["601606 属于铁路→600893 / 钢丝→603969 / 太原重→600072 / 属于专用设备制造业→601106 / "
                     "属于通用设备→600765 / 生产销售日用塑胶制品→002641",
                     "300724 半导体集成电路制造设→002371 / 单晶硅晶体生长炉→300029 / 太阳能光伏专用装备→002006 / "
                     "晶体生长→300316",
                     "300741 食用香精→002568 / 香精→603020", "300664 城市水环境综合治→300388",
                     "002943 控股公司奥瑞德→600666（R2 起头词污染）"],
        "fix": "R3 名称形态约束（3–10 汉字、非动词/连接词/科目词起头、无业务/产品/工序描述词）"
               "+ 与 R1/R2 交叉校验（R3 名不得是本文档 R1/R2 候选名的真片段）+ R2 同受起头词约束",
        "residual": "「太原重」（太原重工截断）形态上无法与真名区分，仍保留 → 已知残余（相邻代码 600072 本身是该件"
                    "同行，pair 落在同行集内但名-码配对错位）；601606 的**码-名错位**（表格列错行）属另一缺陷类，"
                    "本轮未改（登记为已知限制）；**剔除污染名的同时会连带剔除其相邻代码对**（如 300741 华宝股份的"
                    "两条 R3 名-码对：食用香精→002568、香精→603020，其中 603020 是其在网邻居）→ 该件因此由"
                    "「已覆盖」转为「抽到名单但无可映射同行」，与 3 份 CID 恢复件新增的覆盖相抵 → 覆盖率总数"
                    "持平（62/105）",
        "sensitivity": {
            "修正前（v4 交付口径）": {"pair_total": 253, "pair_edge": 137, "overlap_share": 0.541502,
                                     "lift": 25.7681, "diag_pair_total": 245, "diag_pair_edge": 137,
                                     "diag_overlap_share": 0.559184, "diag_lift": 26.6095},
            "复核口径（人工剔 14 名，独立复核 B）": {"pair_total": 246, "pair_edge": 136,
                                                     "overlap_share": 0.552846, "lift": 26.3079,
                                                     "diag_pair_total": 238, "diag_pair_edge": 136,
                                                     "diag_overlap_share": 0.571429, "diag_lift": 27.1922},
            "修正后（本轮规则口径）": {"pair_total": pair_tot, "pair_edge": pair_hit,
                                       "overlap_share": share, "lift": lift,
                                       "diag_pair_total": pair_tot_in, "diag_pair_edge": pair_hit_in,
                                       "diag_overlap_share": share_in, "diag_lift": lift_in},
        },
    }
    # ---------------- 修 4：下界/方向性说明（覆盖率为下界；lift 方向未证，已知偏置指向低估） ----------------
    direction_note = {
        "coverage_lower_bound": "覆盖率为**下界**：分母（试点年 IPO 家数）固定，分子只随抽取/映射成功而增加；"
                                "未定位（含无文本层扫描件）与未抽到的同行只会让覆盖率更低。",
        "lift_direction_unproven": "重合率/lift 的**方向未证**：漏抽同行会同时减少分子（漏掉的同行对）与分母"
                                   "（自报同行对总数），净方向取决于**漏抽同行的落网率**相对全体自报同行落网率的"
                                   "高低，不能先验断言「重合率为下界」。",
        "probe": "方向性检查（本轮，3 份污染/抽漏件）：宽松通道（注册表精确名 + 词边界）补出被漏同行后，"
                 "其网络边落网率与严格通道保留集对照见 anchor_readout.direction_check；"
                 "已知两处偏置（① 2 份无文本层扫描件的同行全漏；② R3 形态约束剔掉的名-码对中含真边）"
                 "方向不一致 → 头条 lift 视为**方向未证**，不宣称下界。",
    }

    # ---------------- 修 3：提取通道计数（按通道类别归并，缓存命中与首次解码同类） ----------------
    channel_counts = {}
    for r in per_firm:
        ch = r.get("channel", "?")
        if ch.startswith("pdfminer+cid_gid"):
            ch = "pdfminer+cid_gid（CID 编码损坏件备选通道）"
        elif "备选通道失败" in ch:
            ch = "cid_recover_failed"
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    # ---------------- 修 4 方向性检查（内联·廉价部分）：被剔名-码对的落网率 vs 保留集 ----------------
    rej = {}
    for x in rejected:
        if x.get("peer") and x["peer"] in firm and x["focal"] in firm:
            fid, j = firm[x["focal"]]["firm_id"], firm[x["peer"]]["firm_id"]
            if fid != j and fid in nodes and j in nodes:
                rej.setdefault((fid, j), x["base"])
    rej_hit = sum(1 for (i, j) in rej if (min(i, j), max(i, j)) in edges)

    def direction_check_cases(pb):
        out = []
        for k, v in ((pb or {}).get("cases") or {}).items():
            if v.get("missed_rate") is not None:
                out.append((k, v))
        return out

    # 宽松通道探针（artifacts/tmp/tier2_prosp_fix/f4_direction.json，由 f4_direction_probe2.py 产出）
    probe_path = os.path.join(ROOT, "artifacts", "tmp", "tier2_prosp_fix", "f4_direction.json")
    probe = None
    try:
        with open(probe_path, encoding="utf-8") as f:
            probe = json.load(f)
    except Exception:
        probe = None
    direction_check = {
        "note": "方向性检查（修 4）：核对「被剔/被漏同行的网络落网率」——若与保留集同量级或更高，"
                "说明抽取遗漏/污染剔除会**改变**（而非仅压低）重合率，故 lift 方向未证。",
        "rejected_pairs_in_network": len(rej), "rejected_pairs_edge": rej_hit,
        "rejected_pairs_edge_rate": (rej_hit / len(rej)) if rej else None,
        "kept_pairs_edge_rate_diag": share_in,
        "kept_pairs_in_network": pair_tot_in, "kept_pairs_edge": pair_hit_in,
        "rejected_examples": sorted({f"{x['focal']}:{x['base']}→{x['peer']}（{x['why']}）"
                                     for x in rejected})[:10],
        "loose_channel_probe": {
            "path": "artifacts/tmp/tier2_prosp_fix/f4_direction.json",
            "method": "宽松通道 = 窗口内注册表精确短名（≥3 汉字）+ 词边界 + 剔自身名；与严格通道保留集比对得"
                      "「被漏同行」，再看其网络边落网率",
            "base_density": (probe or {}).get("base_density"),
            "cases": {k: {"strict_edge/total": v.get("strict_edge/total"),
                          "strict_rate": v.get("strict_rate"),
                          "missed_edge/total": v.get("missed_edge/total"),
                          "missed_rate": v.get("missed_rate"),
                          "missed_names": sorted((v.get("missed_names") or {}).keys())}
                      for k, v in ((probe or {}).get("cases") or {}).items()} if probe else None,
            "reading": "被漏同行的落网率远高于基线密度 "
                       + (f"{(probe or {}).get('base_density', float('nan')):.4f}" if probe else "0.0210")
                       + "，但**相对保留集重合率方向不一**"
                       + ("（"
                          + "；".join(f"{k} = {v['missed_rate']:.4f}"
                                      for k, v in direction_check_cases(probe) if v.get("missed_rate") is not None)
                          + f"，保留集诊断口径 = {share_in:.4f}）" if probe else "")
                       + " → 与独立复核两处探针（均指向低估）合并后，结论 = **方向未证**，不宣称下界",
        },
        "no_text_layer_docs": "300760/300748 为无文本层扫描件（同行全漏，落网率不可测）→ 已知偏置之一",
    }

    result = {
        "report": "第二梯队 #9 路线 A 试点（招股书「同行业可比公司」自报锚，锚 1 / B″11-7）",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "parameter": "prospectus_peer_anchor_pilot（registry 1.5.5，P0 先登记后执行）",
        "scope": {
            "pilot_year": year, "universe": "data/processed/firm_id_map.csv（5,543 只 A 股）",
            "cohort_n": len(cohort), "sample_scope": "全量（不抽样）",
            "network_version_rule": f"试点年 + {args.text_lag} → 文本年 {text_year}（该 focal 上市后第一个"
                                    f"其为网络节点的版本；冻结样本筛选 ma_sample_filter「上市不足 2 年」）",
            "network_version": net_ver, "available_at": f"{text_year + 1}-04-30",
            "download": {"dir": rel(datadir), "n_downloaded": n_downloaded,
                         "status_counts": year_counts,
                         "log": rel(os.path.join(datadir, "download_log.csv")),
                         "summary": rel(os.path.join(datadir, "download_summary.json"))},
            "keyword_set": list(KWS), "window": [WIN_PRE, WIN_POST],
            "extraction_rules": ["R1_full", "R2_name_code", "R3_code_name", "R4_enum", "R5_table_row"],
            "mapping_modes": ["exact_short", "exact_full", "suffix_norm", "adjacent_code"],
            "name_dict": {"firm_universe": len(firm), "short_keys": len(short), "full_keys": len(full),
                          "full_name_build": full_info},
        },
        "rates": {
            "parse_success_rate": parse_rate, "n_located": n_located, "n_downloaded": n_downloaded,
            "mapping_failure_rate": map_fail_rate, "n_unique_names": n_names,
            "n_unmapped_names": len(all_unmapped),
            "coverage_rate": coverage_rate, "n_covered": len(covered), "cohort_n": len(cohort),
            "coverage_rate_in_network": coverage_in_net, "n_network_nodes": n_net_nodes,
            "thresholds": {"parse_min": 0.80, "map_fail_max": 0.20, "coverage_min": 0.50},
            "stop_triggered": bool(stop), "stop_reasons": stop,
            "coverage_breakdown": cov_breakdown,
            "mapping_mode_totals": mode_totals,
            "peers_per_firm_mean": (sum(peers_per_firm) / len(peers_per_firm)) if peers_per_firm else None,
            "parse_iterations": parse_iterations,
            "failure_decomposition": failure_decomposition,
            "extraction_channels": channel_counts,
        },
        "anchor_readout": {
            "definition": "lift = P(自报同行 ∈ 网络对手 | 自报同行) ÷ P(随机 universe 公司 ∈ 网络对手)"
                          "（分母 = 该网络年无向对密度 d = E/(N(N−1)/2)）",
            "overlap_share_raw": share, "pair_total": pair_tot, "pair_edge": pair_hit,
            "base_density": base_density, "lift": lift,
            "diagnostic_node_restricted": {"overlap_share_raw": share_in, "pair_total": pair_tot_in,
                                           "pair_edge": pair_hit_in, "lift": lift_in,
                                           "note": "诊断变体：只统计两端都在该网络年节点集内的自报同行对"},
            "per_firm_overlap_stats": ov_stats,
            "network": {"text_year": text_year, "version": net_ver, "N_nodes": N,
                        "E_undirected": E},
            "direction_check": direction_check,
        },
        "extraction_pollution": r3_pollution,
        "direction_note": direction_note,
        "failure_cases_sample": fail_cases,
        "unmapped_top30": [{"name": k, "n_firms": v} for k, v in unmapped_top],
        "known_limitations": [
            "**覆盖率为下界**（分母固定、分子只增）；**重合率/lift 方向未证**——漏抽同行同时减少分子与分母，"
            "净方向取决于漏抽同行的落网率（方向性检查见 anchor_readout.direction_check 与 §2.2），"
            "不得写成「重合率为下界」。",
            "2 份招股书（300760 迈瑞医疗、300748 金力永磁）为**无文本层扫描件**（620/413 页仅 2,480/948 字），"
            "pymupdf 与 pdfminer 均无文本层 → 未进入解析分母；全量需补 OCR 通道。",
            "3 份招股书（300737 科顺股份、603059 倍加洁、603301 振德医疗）为**文本层 CID 编码损坏**"
            "（无 ToUnicode CMap，pymupdf 读出 chr(cid) 乱码、「公司/发行人/万元」计数 = 0）；本轮经 pdfminer.six "
            "+ SimSun GID 反查备选通道恢复（确定性字符还原、非 OCR），已进入解析/抽取/映射口径。",
            "单年（2018）、单网络版本（pm_ma_ewcos_teacher_v1_2020）试点；lift 不能外推为全期结论。",
            "项目 firm 名单为 2026 当前简称，招股书用 2018 当时简称；已用年报头部「简称+代码」（标签/顺序/冒号变体、"
            "去 ST 前缀）、更名披露与释义缩写四条代码锚定来源补历史简称，残余漂移仍计入映射失败。",
            "已知残余（本轮登记）：① 「太原重」（太原重工截断）形态上与真名不可分，仍经相邻代码映射；"
            "② 601606 等表格列错行造成的**码-名错位**（名与码各自都在同行集内、但配对错位）未在本轮修复。",
            "锚只作外部一致性读数，**不进入面板/回归、不改主口径**；未完成全量前「外部锚」条件仍未满足。",
        ],
        "per_firm": per_firm,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "firm_id", "name", "ipo_date", "board", "status", "n_kw_hits",
                    "first_hit_page", "n_unique_names", "n_mapped", "n_unmapped", "network_node",
                    "n_pairs", "n_pairs_edge", "overlap_share", "peers"])
        for r in detail:
            w.writerow([r.get("code"), r.get("firm_id"), r.get("name"), r.get("ipo_date"),
                        r.get("board"), r.get("status"), r.get("n_kw_hits", ""),
                        r.get("first_hit_page", ""), r.get("n_unique_names", ""), r.get("n_mapped", ""),
                        r.get("n_unmapped", ""), r.get("network_node", ""), r.get("n_pairs", ""),
                        r.get("n_pairs_edge", ""), r.get("overlap_share", ""),
                        "|".join(r.get("peers", []) or [])])
    write_md(result, out_md)
    print(f"[P-6] 报告 → {rel(out)} / {rel(out_md)}；明细 → {rel(detail_csv)}", flush=True)
    print(f"[P-6] 三项率 = 解析 {parse_rate:.4f}（{n_located}/{n_downloaded}）/ 映射失败 "
          f"{map_fail_rate:.4f}（{len(all_unmapped)}/{n_names}）/ 覆盖率 {coverage_rate:.4f}"
          f"（{len(covered)}/{len(cohort)}）", flush=True)
    print(f"[P-6] 锚读数 = 重合率 {share:.6f}（{pair_hit}/{pair_tot}）vs 基线 {base_density:.6f} "
          f"→ lift {lift:.4f}（诊断·限在网 {lift_in:.4f}）", flush=True)
    print(f"[P-6] 停机 = {'触发: ' + '; '.join(stop) if stop else '未触发'}", flush=True)
    with open(out.replace(".json", "_run.log"), "w", encoding="utf-8") as f:
        f.write(json.dumps({k: result[k] for k in ("run_at", "rates", "anchor_readout", "scope")},
                           ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
