# -*- coding: utf-8 -*-
"""生成 analyst-network 共同分析师网络汇报 PPTX（16:9，8 页，可编辑版）。

用法: python scripts/make_analyst_network_brief_pptx.py
输出: deliverables/presentation/peermax_analyst_report_20260901.pptx
数字来源: reports/execution/analyst_report.md v2.0
v1.1: 字号整体调大 1-2pt；修复第 7 页右栏遗留项溢出卡片、第 4 页数字卡 note 位置。
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
import os

# ---------------------------------------------------------------- 设计常量
NAVY = RGBColor(0x1F, 0x38, 0x64)      # 主色（深蓝）
NAVY2 = RGBColor(0x2E, 0x4D, 0x7E)     # 次级蓝
ORANGE = RGBColor(0xC5, 0x5A, 0x11)    # 强调（数据橙）
RED = RGBColor(0xC0, 0x00, 0x00)       # 警示（待办）
GRAY = RGBColor(0x59, 0x59, 0x59)      # 正文
LGRAY = RGBColor(0xF2, 0xF2, 0xF2)     # 卡片浅底
MGRAY = RGBColor(0xD9, 0xD9, 0xD9)     # 分隔线
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

EMU_IN = 914400
PAGE_W, PAGE_H = 13.333, 7.5


def set_font(run, size, bold=False, color=GRAY, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    latin = rPr.find(qn("a:latin"))
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        if latin is not None:
            latin.addnext(ea)
        else:
            rPr.append(ea)
    ea.set("typeface", name)


def set_cell_border(cell, color="BFBFBF", width=6350):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for e in ("L", "R", "T", "B"):
        tag = "a:ln" + e
        ln = tcPr.find(qn(tag))
        if ln is None:
            ln = tcPr.makeelement(qn(tag), {})
            tcPr.append(ln)
        ln.set("w", str(width))
        fill = ln.makeelement(qn("a:solidFill"), {})
        clr = fill.makeelement(qn("a:srgbClr"), {"val": color})
        fill.append(clr)
        ln.append(fill)


def add_text(slide, x, y, w, h, text, size=14, bold=False, color=GRAY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = ln
        set_font(run, size, bold, color)
    return tb


def add_shape(slide, shape_type, x, y, w, h, fill=None, line_color=None,
              line_w=0.75, text=None, tsize=14, tbold=False, tcolor=WHITE,
              talign=PP_ALIGN.CENTER):
    sp = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line_color is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line_color
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    if text is not None:
        tf = sp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, ln in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = talign
            run = p.add_run()
            run.text = ln
            set_font(run, tsize, tbold, tcolor)
    return sp


def header(slide, title, page_no, subtitle=None):
    add_shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, 0.95, fill=NAVY)
    add_text(slide, 0.55, 0.13, 9.6, 0.7, title, size=26, bold=True, color=WHITE,
             anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(slide, 9.7, 0.13, 3.1, 0.7, subtitle, size=12, color=RGBColor(0xBD, 0xC7, 0xDC),
                 align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, 12.5, 7.08, 0.7, 0.35, f"{page_no} / 8", size=10, color=MGRAY,
             align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, title, body, tcolor=NAVY, bcolor=GRAY, tsize=14, bsize=12.5,
         fill=WHITE, line=MGRAY, title_fill=None):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill,
              line_color=line, line_w=0.75)
    if title_fill is not None:
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 0.44, fill=title_fill)
    add_text(slide, x + 0.16, y + 0.06, w - 0.32, 0.36, title, size=tsize, bold=True,
             color=tcolor, anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, x + 0.16, y + 0.52, w - 0.32, h - 0.6, body, size=bsize, color=bcolor,
             line_spacing=1.15)


# ---------------------------------------------------------------- 页面构建
def slide_cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_shape(s, MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, PAGE_H, fill=NAVY)
    add_shape(s, MSO_SHAPE.RECTANGLE, 0, 4.9, PAGE_W, 0.06, fill=ORANGE)
    add_text(s, 1.0, 1.9, 11.3, 1.0, "analyst-network 共同分析师覆盖网络", size=46, bold=True, color=WHITE)
    add_text(s, 1.0, 3.0, 11.3, 0.7, "文献研读 · A 股构造 · 复现对照 · 六维证据",
             size=22, color=RGBColor(0xBD, 0xC7, 0xDC))
    add_text(s, 1.0, 5.2, 11.3, 0.55,
             "4,972 万条无向边  ·  241 个月度网络（全期 2006-2025）  ·  参数 5/5 冻结  ·  复现对照通过",
             size=16, bold=True, color=ORANGE)
    add_text(s, 1.0, 6.6, 11.3, 0.4, "PeerMAX 项目 · 2026-08-29 · 详细版 analyst_report.md v2.0",
             size=12.5, color=RGBColor(0x9D, 0xAB, 0xC4))


def slide_position(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "项目定位：analyst-network 在执行链中的位置", 2)
    blocks = [
        ("S0", "数据基座", NAVY, "收益 / 公告 / 文本"),
        ("daily-panel", "面板与因子", NAVY, "factor 自建 FF4 并行"),
        ("analyst-network", "分析师网络", ORANGE, "本次汇报 · 已完成"),
        ("network", "产品市场网络", MGRAY, "等老师数据"),
        ("panel", "PeerMAX 面板", NAVY, "依赖产品市场 network"),
    ]
    x, y, bw, bh, gap = 0.55, 2.0, 2.28, 1.3, 0.16
    arrow_w = (PAGE_W - 2 * x - 5 * bw - 4 * gap) / 4
    for i, (code, name, fill, note) in enumerate(blocks):
        cx = x + i * (bw + gap + arrow_w)
        add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, cx, y, bw, bh, fill=fill)
        add_text(s, cx, y + 0.18, bw, 0.55, code, size=26, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER)
        add_text(s, cx, y + 0.72, bw, 0.45, name, size=13.5, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER)
        if i < 4:
            ax = cx + bw + gap / 2
            add_shape(s, MSO_SHAPE.RIGHT_ARROW, ax, y + 0.48, arrow_w, 0.38,
                      fill=MGRAY)
    for i, (code, name, fill, note) in enumerate(blocks):
        cx = x + i * (bw + gap + arrow_w)
        add_text(s, cx, y + bh + 0.14, bw, 0.6, note, size=12, color=GRAY,
                 align=PP_ALIGN.CENTER)
    add_text(s, 0.55, 4.35, 12.2, 1.5,
             "analyst-network 是辅助网络（alternative network）：给主线提供控制变量、对照检验与机制证据，"
             "不替代 network 产品市场网络，不进入 headline PeerMAXraw。\n"
             "主线当前状态：S0-factor 完成，network 文本收口三项（2017 缺口 640 / 错位 1,775 / 408 假边）已执行完，"
             "只等产品市场数据；panel 待 network 数据到位后开跑。",
             size=14, color=GRAY, line_spacing=1.35)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 5.95, 12.2, 0.8, fill=LGRAY,
              line_color=MGRAY)
    add_text(s, 0.75, 6.1, 11.8, 0.55,
             "analyst-network 成果：5 项参数 FROZEN · 契约测试 7/7 · 分析师覆盖已写入主线控制变量表（49.7 万行）",
             size=14, bold=True, color=NAVY)


def slide_literature(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "文献研读：三方口径对照", 3)
    rows = [
        ("维度", "本项目 analyst-network", "AH · 美股（IBES）", "Long · A 股（CSMAR）"),
        ("连接时点", "月末快照", "每月末", "每月末"),
        ("覆盖窗口", "12 个月", "12 个月", "12 个月"),
        ("预测期限定", "FY1 / FY2（A 股财年=日历年）", "FY1 / FY2", "不限定"),
        ("权重（计数+行归一化）", "n_ij 线性（=AH 式1）", "n_ij 线性（式1）", "ln(n_ij+1) 对数（式2）"),
        ("数据源 / 样本期", "CNRDS · 2006-2025（全期）", "IBES · 1983-2015", "CSMAR · 2004-2022"),
    ]
    tbl_shape = s.shapes.add_table(len(rows), 4, Inches(0.55), Inches(1.25),
                                   Inches(12.2), Inches(3.8))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(3.6)
    tbl.columns[2].width = Inches(3.2)
    tbl.columns[3].width = Inches(3.2)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = val
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
                set_font(run, 14, True, WHITE)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else LGRAY
                set_font(run, 13, c == 0, NAVY if c == 0 else GRAY)
            set_cell_border(cell)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 5.3, 12.2, 1.05, fill=LGRAY,
              line_color=MGRAY)
    add_text(s, 0.75, 5.52, 11.8, 0.6,
             "分工一句话：网络层照 AH（连接 / 窗口 / 权重），应用层走 Long（NPM / 五分组 / FM 回归），"
             "A 股本地化只动 FY 规则与数据源。",
             size=15, bold=True, color=NAVY)
    add_text(s, 0.55, 6.6, 12.2, 0.5,
             "锚点：AH 网络指纹=约 70% 连接仅 1 名共同分析师 · Long 数值锚点=NPM 1.28%/月 t=5.20、CF α 0.97%/t=3.52",
             size=12.5, color=GRAY)


def slide_network(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "网络构建：数据漏斗与网络指纹", 4)
    steps = [
        ("2,922,904", "原始行", "CNRDS 6 分片，0 坏行"),
        ("1,815,139", "覆盖报告", "FY1/FY2 行级过滤先于报告级去重"),
        ("241", "月度网络", "2005-12 ~ 2025-12 月末快照（全期）"),
        ("49,719,505", "无向边", "5,087 公司 · 12,638 分析师"),
    ]
    x, y, w, gap = 0.55, 1.3, 2.9, 0.22
    for i, (num, label, note) in enumerate(steps):
        cx = x + i * (w + gap)
        add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, cx, y, w, 1.75, fill=NAVY)
        add_text(s, cx, y + 0.2, w, 0.85, num, size=30, bold=True, color=ORANGE,
                 align=PP_ALIGN.CENTER)
        add_text(s, cx, y + 1.05, w, 0.5, label, size=15, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER)
        if i < 3:
            add_shape(s, MSO_SHAPE.RIGHT_ARROW, cx + w + gap / 2, y + 0.68, gap, 0.34,
                      fill=MGRAY)
        add_text(s, cx, y + 1.86, w, 0.5, note, size=11.5, color=GRAY,
                 align=PP_ALIGN.CENTER)
    add_text(s, 0.55, 3.78, 12.2, 0.4,
             "防未来信息：只用报告发布日 ≤ 月末快照的记录；实际公告日（Anndats_act）一律不进网络。",
             size=13, color=GRAY)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 4.3, 5.9, 2.3, fill=LGRAY, line_color=MGRAY)
    add_text(s, 0.75, 4.45, 5.5, 0.45, "网络指纹（与美股同向）", size=15, bold=True, color=NAVY)
    add_text(s, 0.75, 4.95, 5.5, 1.6,
             "共同分析师人数中位：1.31\n弱边（common=1）占比 52.6%\n（AH 约 70%、蒋约 61%：同以弱连接为主，A 股覆盖更集中略低）\n连接数中位 161 · p90 376（美股均值 86）",
             size=13, color=GRAY, line_spacing=1.25)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.75, 4.3, 6.0, 2.3, fill=LGRAY, line_color=MGRAY)
    add_text(s, 6.95, 4.45, 5.6, 0.45, "质量验收（全部通过）", size=15, bold=True, color=NAVY)
    add_text(s, 6.95, 4.95, 5.6, 1.6,
             "唯一键重复 0 · 自环 0 · (i,j)/(j,i) 不对称 0\n"
             "权重和误差 ≤ 2.9e-6（阈值 1e-4）· 契约测试 7/7（版本自适应+manifest 守卫）\n"
             "无连接公司月均 2.1 家（标记 no-network-support，不赋零）",
             size=13.5, color=GRAY, line_spacing=1.3)


def slide_replication(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "复现结果：我们的网络 vs 已发表论文", 5)
    rows = [
        ("检验（全期 2006-2025，241 月）", "本项目", "论文锚点", "判定"),
        ("CF 多空 EW", "0.85%/月 · t=2.93", "Jansen 0.46%/t=1.54\n（IBES，同口径）", "显著 ✓"),
        ("CF 多空 VW", "0.90%/月 · t=2.35", "蒋丹凌 VW 0.895%/t2.71", "转正显著 ✓"),
        ("CF h12 十二个月动量", "0.40%/月 · 五分位单调", "蒋丹凌 0.565***（持续 12 月）", "单调复现"),
        ("CF + 对数权重 + 五因子 α", "1.97%/月 · t=7.13（全期 227 月）", "AH 2.10%/t=11.88（同框架，≈94%）", "同框架 ✓"),
        ("NPM 多空 EW（净同行动量）", "1.77%/月 · t=7.54", "Long 1.28%/t=5.20", "raw 口径超锚点；后复权可比口径 1.26%/t5.62 与锚点持平"),
        ("NPM 强于 CF", "1.77 > 0.85", "论文核心结论", "复现 ✓"),
    ]
    tbl_shape = s.shapes.add_table(len(rows), 4, Inches(0.55), Inches(1.25),
                                   Inches(12.2), Inches(3.9))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(3.3)
    tbl.columns[1].width = Inches(3.7)
    tbl.columns[2].width = Inches(3.0)
    tbl.columns[3].width = Inches(2.2)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = val
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
                set_font(run, 14, True, WHITE)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else LGRAY
                bold = c == 0 or (r == 3 and c == 3) or (r == 1 and c == 1)
                set_font(run, 13, bold, NAVY if c == 0 else GRAY)
            set_cell_border(cell)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 5.4, 12.2, 1.15, fill=LGRAY,
              line_color=MGRAY)
    add_text(s, 0.75, 5.57, 11.8, 0.85,
             "口径注：全期因子面板已建（REV 因子 2000-03 起、FF4 2007-02 起），五因子 α 样本 227 个月；Long CF α 0.97% 为 VW 六因子 spanning α，仅方向参考\n"
             "覆盖口径验证（9K）：另建不限 FY 的对照网络（Long 口径=本数据源蒋口径等值，预测表无'纯报告不含预测'记录），总边仅 +0.047%、NPM 1.77%/t7.54 完全相同——raw 口径下的超锚点不依赖覆盖判定；可比口径（后复权）为持平\n"
             "样本扩展：2018-2025 短样本 VW 不显著/h12 不显著 → 扩至 2006-2025 后 VW 转正显著、h12 单调（样本期效应，非网络问题）",
             size=13.5, color=GRAY, line_spacing=1.35)
    add_text(s, 0.55, 6.68, 12.2, 0.35,
             "二阶连接：hi-lo 诊断 0.45%/t=2.50，但正式规格控制一阶 CF 后不显著（9M，t=0.74，与 AH Table 8 不同、如实报告）· CF 组合反转载荷 −60.6%（全期，印证 NPM 动机）",
             size=12.5, color=GRAY)


def slide_evidence(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "六维证据包（Step 9，Drive 数据）", 6)
    cards = [
        ("① 机制 · 基本面联动", "连接公司营收/净利增速与焦点联动（全期 19 年）\n营收 β=0.061 · t=8.42\n净利 β=0.172 · t=13.14\n（对标 AH Table 2）", ORANGE),
        ("② 独特 · 行业对照+吸收", "同行业边 lift 1.42→6.48（随覆盖密度上升）\nCF-行业动量逐月相关 241 月均值 0.62\n9H：同入后行业动量塌缩 0.091→0.005、CF t=6.69\n→ CF 吸收行业动量（显著，对标蒋表 5）", ORANGE),
        ("③ 回归 · Fama-MacBeth", "NPM β=0.1242 · t=5.89（NW1 5.94，全期 241 月）\n控制 size/BM/反转/动量/换手/流动性\n→ 预测力非已知因子代理\n（对标 Long Table 5）", ORANGE),
        ("④ 稳健 · ST 剔除", "全期 NPM 1.77→1.79%（t=7.61）\n剔除后略升 → 非 ST 驱动", NAVY),
        ("⑤ 稳健 · 大小盘 + VW", "全期大小盘全显著：小盘 2.22/t8.82\n中盘 1.29/t4.77 · 大盘 1.24/t4.61\nVW（Drive 市值）1.02% · t=3.39 转正", NAVY),
        ("⑥ 口径 · 后复权 + rf", "全期后复权：CF 0.85→0.87 稳\nNPM 1.77→1.26（差分收窄）\n方向/单调/NPM>CF 全稳；主线换口径待拍板", NAVY),
    ]
    x0, y0, cw, ch, gx, gy = 0.55, 1.25, 4.0, 2.6, 0.1, 0.2
    for i, (title, body, tcolor) in enumerate(cards):
        r, c = divmod(i, 3)
        cx = x0 + c * (cw + gx)
        cy = y0 + r * (ch + gy)
        add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, cw, ch, fill=LGRAY, line_color=MGRAY)
        add_text(s, cx + 0.18, cy + 0.12, cw - 0.36, 0.42, title, size=14, bold=True,
                 color=tcolor)
        add_text(s, cx + 0.18, cy + 0.62, cw - 0.36, ch - 0.7, body, size=12.5,
                 color=GRAY, line_spacing=1.25)


def slide_governance(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "治理与遗留", 7)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 1.25, 6.0, 5.6, fill=LGRAY, line_color=MGRAY)
    add_text(s, 0.75, 1.42, 5.6, 0.45, "治理（已完成）", size=16, bold=True, color=NAVY)
    gov = [
        "5 项参数 FROZEN：覆盖窗口 12 月 / FY1/FY2 规则 / 月末快照 / 权重精度 1e-4 / 权重规则",
        "依据：论文口径逐条对照 + 三轮对抗审查",
        "v1→v2 缺陷修复留痕（FY 过滤顺序错误：误杀 17.0%、丢 8.2% → 修复重建全量归档）",
        "契约测试 tests/test_analyst_contracts.py 7/7（版本自适应，含 manifest 重复键守卫）",
        "敏感性矩阵全执行：权重 ln / 门槛 ≥2/≥5 / 窗口 1-12 月 / 身份冲突 / 券商 / 样本",
        "下游：AnalystCoverage 写入主线控制变量表 497,211 行（此前为空列）",
    ]
    body = "\n".join(f"▸ {g}" for g in gov)
    add_text(s, 0.75, 2.0, 5.6, 4.6, body, size=13, color=GRAY, line_spacing=1.4)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.85, 1.25, 5.9, 5.6, fill=WHITE,
              line_color=MGRAY)
    add_text(s, 7.05, 1.42, 5.5, 0.45, "遗留（3 项）", size=16, bold=True, color=RED)
    left = [
        ("① 三人独立复现", "等同学各自实现后三方比对：\n节点/边数/degree/common-count/权重分布/CF-NPM 结果\n（外部依赖）", "外部依赖"),
        ("② 收益主口径", "主线现用 raw 价（不含分红）；\n后复权敏感性已出（CF 0.83/NPM 0.75）\n（换主线需老师拍板）", "请您决策"),
        ("③ network 产品市场数据", "文本收口三项已执行完，\n只等词库/相似度矩阵/申万数据到位开跑\n（外部依赖）", "外部依赖"),
    ]
    ly = 2.0
    for title, body, tag in left:
        add_text(s, 7.05, ly, 5.5, 0.4, title, size=14.5, bold=True, color=GRAY)
        add_text(s, 7.05, ly + 0.42, 5.5, 0.85, body, size=12.5, color=GRAY,
                 line_spacing=1.2)
        ly += 1.4


def slide_close(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "收尾：对主线的意义与下一步", 8)
    add_text(s, 0.55, 1.3, 12.2, 0.5, "analyst-network 对 PeerMAX 的三层价值", size=17, bold=True, color=NAVY)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.55, 1.9, 3.9, 1.75, fill=LGRAY, line_color=MGRAY)
    add_text(s, 0.73, 2.06, 3.55, 0.45, "控制变量", size=15, bold=True, color=NAVY)
    add_text(s, 0.73, 2.55, 3.55, 1.0, "分析师覆盖已进主线控制变量表，定价 gate / 机制层 回归可用（此前为空列）", size=12.5, color=GRAY)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 4.7, 1.9, 3.9, 1.75, fill=LGRAY, line_color=MGRAY)
    add_text(s, 4.88, 2.06, 3.55, 0.45, "对照检验", size=15, bold=True, color=NAVY)
    add_text(s, 4.88, 2.55, 3.55, 1.0, "证明主线结果不是任何网络都能跑出来的（机制层对照）", size=12.5, color=GRAY)
    add_shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 8.85, 1.9, 3.9, 1.75, fill=LGRAY, line_color=MGRAY)
    add_text(s, 9.03, 2.06, 3.55, 0.45, "机制证据", size=15, bold=True, color=NAVY)
    add_text(s, 9.03, 2.55, 3.55, 1.0, "信息沿网络扩散的机制路径（S11 机制层）", size=12.5, color=GRAY)
    add_text(s, 0.55, 3.95, 12.2, 0.5, "方法学复用", size=17, bold=True, color=NAVY)
    add_text(s, 0.55, 4.45, 12.2, 1.1,
             "论文口径核对 → 网络构造 → 参数冻结 → 复现对照 → 扩展检验 的整条流水线在 A 股跑通，"
             "network 产品市场网络直接复用这套方法与检验范式。",
             size=14, color=GRAY, line_spacing=1.35)
    add_text(s, 0.55, 5.75, 12.2, 0.5, "下一步", size=17, bold=True, color=NAVY)
    add_text(s, 0.55, 6.25, 12.2, 0.8,
             "三人复现比对通过 → 数据门 G2 转「可得」；收益主口径待老师拍板；network 数据到位即开跑产品市场网络。",
             size=14, color=GRAY, line_spacing=1.35)


def main():
    prs = Presentation()
    prs.slide_width = Emu(int(PAGE_W * EMU_IN))
    prs.slide_height = Emu(int(PAGE_H * EMU_IN))
    slide_cover(prs)
    slide_position(prs)
    slide_literature(prs)
    slide_network(prs)
    slide_replication(prs)
    slide_evidence(prs)
    slide_governance(prs)
    slide_close(prs)
    out = os.path.join("deliverables", "presentation", "peermax_analyst_report_20260901.pptx")
    prs.save(out)
    print(f"saved: {out} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
