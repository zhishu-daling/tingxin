"""
听心 · ECG GUI v2 — 波形+频率+情绪+可播放
"""
import sys, io, os, time, threading, tempfile, json
import ctypes

import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
import numpy as np
import soundfile as sf
from core.stream_pipeline import StreamPipeline
from core.preprocess import load_audio, SAMPLE_RATE

BASE = Path(__file__).parent; OUTPUT_DIR = BASE / "output"
SPK_COLORS = {"SPK1":"#58a6ff","SPK2":"#56d364","SPK3":"#e3b341",
              "SPK4":"#f85149","SPK5":"#d2a8ff","UNKNOWN":"#484f58"}

def classify_emotion(feat):
    """基于声学特征的简单情绪分类"""
    if not feat: return "—", "#8b949e"
    energy = feat.get("energy_mean", 0)
    pitch_std = feat.get("pitch_std", 0)
    if energy > 0.15 and pitch_std > 50: return "激动", "#f85149"
    if energy > 0.10: return "平和", "#d29922"
    if pitch_std < 15: return "低沉", "#58a6ff"
    return "正常", "#3fb950"

class TingXin:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("听心 ECG")
        self.root.geometry("1100x800")
        self.root.configure(bg="#0d1117")
        self._audio = None; self._sr = SAMPLE_RATE; self._segments = []
        self._current_file = None; self._running = False; self._pipe = None
        self._play_thread = None
        self._batch_files = []; self._batch_meeting_mode = False
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # 顶栏
        m = tk.Frame(self.root, bg="#161b22", height=40)
        m.pack(fill="x"); m.pack_propagate(False)
        tk.Label(m, text=" 听心", font=("Segoe UI",15,"bold"),bg="#161b22",fg="#e6edf3").pack(side="left",padx=12)
        self.lbl_stat = tk.Label(m, text="就绪", font=("Segoe UI",10),bg="#161b22",fg="#8b949e")
        self.lbl_stat.pack(side="right",padx=12)

        # 波形图
        self.wave = tk.Canvas(self.root, height=160, bg="#0d1117", highlightthickness=0)
        self.wave.pack(fill="x", padx=4, pady=(4,0))
        self.wave.bind("<Button-1>", self._on_click_wave)

        # 信息栏
        info = tk.Frame(self.root, bg="#0d1117")
        info.pack(fill="x", padx=12, pady=(2,0))
        self.lbl_time = tk.Label(info, text="0:00", font=("Consolas",10),bg="#0d1117",fg="#484f58")
        self.lbl_time.pack(side="left")
        self.lbl_prog = tk.Label(info, text="", font=("Consolas",10),bg="#0d1117",fg="#d29922")
        self.lbl_prog.pack(side="right")

        # 进度条
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", background="#1f6feb", troughcolor="#21262d",
                        bordercolor="#0d1117", lightcolor="#1f6feb", darkcolor="#1f6feb")
        self.progress = ttk.Progressbar(info, length=200, mode='determinate')
        self.progress.pack(side="right", padx=(8,0))

        # 按钮
        bar = tk.Frame(self.root, bg="#0d1117")
        bar.pack(fill="x", padx=12, pady=(4,2))
        tk.Button(bar, text="选择录音", font=("Segoe UI",10),bg="#1f6feb",fg="white",
                  relief="flat",padx=14,pady=4,command=self._pick).pack(side="left",padx=(0,4))
        self.btn_start = tk.Button(bar, text="开始", font=("Segoe UI",10),bg="#3fb950",
                                   fg="white",relief="flat",padx=14,pady=4,state="disabled",command=self._start)
        self.btn_start.pack(side="left",padx=4)
        self.btn_export = tk.Button(bar, text="导出报告", font=("Segoe UI",10),bg="#21262d",
                                    fg="#58a6ff",relief="flat",padx=14,pady=4,state="disabled",
                                    command=self._export_report)
        self.btn_export.pack(side="left",padx=4)
        self.btn_batch = tk.Button(bar, text="批量处理", font=("Segoe UI",10),bg="#21262d",
                                    fg="#e3b341",relief="flat",padx=14,pady=4,
                                    command=self._batch_pick)
        self.btn_batch.pack(side="left",padx=4)
        self.btn_meeting = tk.Button(bar, text="会议纪要", font=("Segoe UI",10),bg="#f0883e",
                                     fg="white",relief="flat",padx=14,pady=4,
                                     command=self._meeting_direct)
        self.btn_meeting.pack(side="left",padx=4)
        self.btn_batch_meeting = tk.Button(bar, text="批量会议", font=("Segoe UI",10),bg="#21262d",
                                           fg="#f0883e",relief="flat",padx=14,pady=4,
                                           command=self._batch_meeting_pick)
        self.btn_batch_meeting.pack(side="left",padx=4)
        self.btn_music = tk.Button(bar, text="音乐分析", font=("Segoe UI",9),bg="#c9abff",
                                   fg="#0d1117",relief="flat",padx=10,pady=3,
                                   command=self._music_analyze)
        self.btn_music.pack(side="left",padx=2)
        self.btn_batch_music = tk.Button(bar, text="批量音乐", font=("Segoe UI",9),bg="#21262d",
                                         fg="#c9abff",relief="flat",padx=10,pady=3,
                                         command=self._batch_music_pick)
        self.btn_batch_music.pack(side="left",padx=2)
        self.btn_call = tk.Button(bar, text="通话分析", font=("Segoe UI",9),bg="#ff7b72",
                                  fg="white",relief="flat",padx=10,pady=3,
                                  command=self._call_analyze)
        self.btn_call.pack(side="left",padx=2)
        self.btn_batch_call = tk.Button(bar, text="批量话务", font=("Segoe UI",9),bg="#21262d",
                                        fg="#ff7b72",relief="flat",padx=10,pady=3,
                                        command=self._batch_call_pick)
        self.btn_batch_call.pack(side="left",padx=2)

        # 时间线（Text widget，轻量）
        self.tl = None
        tl = tk.Frame(self.root, bg="#161b22", bd=1, relief="solid",
                      highlightbackground="#30363d",highlightthickness=1)
        tl.pack(fill="both",expand=True,padx=4,pady=(4,8))
        tk.Label(tl, text="  时间线", font=("Segoe UI",10,"bold"),
                 bg="#161b22",fg="#e6edf3").pack(anchor="w",padx=8,pady=(4,0))
        self.tl_text = tk.Text(tl, bg="#0d1117", fg="#c9d1d9", font=("Consolas",10),
                               relief="flat", highlightthickness=0, cursor="hand2",
                               wrap="none", padx=6, pady=4)
        scroll = tk.Scrollbar(tl, orient="vertical", command=self.tl_text.yview)
        self.tl_text.configure(yscrollcommand=scroll.set)
        self.tl_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        # tag 样式
        for i in range(1,6):
            self.tl_text.tag_configure(f"spk{i}", foreground=SPK_COLORS.get(f"SPK{i}"))
        self.tl_text.tag_configure("ts", foreground="#484f58")
        self.tl_text.tag_configure("bold", font=("Consolas",10,"bold"))

    def _pick(self):
        p = filedialog.askopenfilename(filetypes=[("音频","*.mp3 *.wav *.m4a *.flac *.ogg")])
        if p: self._current_file = p; self.lbl_stat.config(text=f"已选: {os.path.basename(p)}"); self.btn_start.config(state="normal")

    def _batch_pick(self):
        d = filedialog.askdirectory()
        if not d: return
        exts = ('.mp3','.wav','.m4a','.flac','.ogg','.aac','.wma')
        files = []
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(exts):
                files.append(os.path.join(d, f))
        if not files:
            self.lbl_stat.config(text="所选文件夹无音频文件"); return
        self._batch_files = files
        self.lbl_stat.config(text=f"批量 {len(files)}个文件")
        self.btn_start.config(state="disabled")
        self.btn_batch.config(state="disabled",text="批量中...")
        threading.Thread(target=self._batch_process, daemon=True).start()

    def _batch_process(self):
        total = len(self._batch_files)
        ok, fail = 0, 0
        for idx, fp in enumerate(self._batch_files):
            fname = os.path.basename(fp)
            self.root.after(0, lambda i=idx+1,t=total,n=fname:
                self.lbl_stat.config(text=f"批量 {i}/{t}: {n}"))
            self.root.after(0, lambda: self.lbl_prog.config(text=f"{idx+1}/{total}"))
            try:
                self._process_single_batch(fp, idx+1, total)
                ok += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                fail += 1
                self.root.after(0, lambda f=fp,e=e:
                    self.lbl_stat.config(text=f"错误: {os.path.basename(f)} - {e}"))
        self.root.after(0, lambda: self.lbl_stat.config(text=f"批量完成: {ok}成功/{fail}失败"))
        self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
        self._running = False
        self._batch_meeting_mode = False
        self.root.after(0, lambda: self.btn_batch.config(state="normal",text="批量处理"))
        self.root.after(0, lambda: self.btn_export.config(state="normal"))

    def _start(self):
        if not self._current_file or self._running: return
        self._running = True; self._segments = []
        self.btn_start.config(state="disabled",text="处理中")
        self.tl_text.config(state="normal"); self.tl_text.delete("1.0","end"); self.tl_text.config(state="disabled")
        self.wave.delete("all"); self.lbl_prog.config(text="加载...")
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        try:
            self._audio = load_audio(self._current_file, sr=self._sr)
            total = len(self._audio)/self._sr
            self.root.after(0, lambda: self.lbl_time.config(text=f"0:00 / {int(total)//60}:{int(total)%60:02d}"))
            self.root.after(0, lambda: self.lbl_stat.config(text="处理中..."))

            pipe = StreamPipeline(self._current_file)
            self._pipe = pipe

            def cb(r):
                # MEITR 分析进度
                if r.get("analyzing"):
                    pct = int(r.get("analysis_progress",0)*100)
                    self.root.after(0, lambda p=pct: self.lbl_prog.config(text=f"分析中 {p}%"))
                    self.root.after(0, lambda p=pct: self.progress.config(value=p))
                    return

                segs = r.get("new_segments",[])
                if segs:
                    self._segments.extend(segs)
                    n = len(self._segments)
                    self.root.after(0, lambda n=n,p=int(r.get("progress",0)*100): self.lbl_prog.config(text=f"{p}% · {n}句") or self.progress.config(value=p))
                    self.root.after(0, lambda: self._add_segments(segs))
                    self.root.after(0, lambda: self._draw_all())
                ref = r.get("refined")
                if ref:
                    self._segments = ref
                    spks = sorted(set(s["speaker"] for s in ref))
                    self.root.after(0, lambda: self._rebuild(ref))
                    self.root.after(0, lambda: self._draw_all())
                    self.root.after(0, lambda: self.lbl_stat.config(text=f"{len(spks)}人 · {len(ref)}句"))

            result = pipe.run(callback=cb, output_dir=str(OUTPUT_DIR))
            spk = result["num_speakers"]
            self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
            self.root.after(0, lambda: self.lbl_stat.config(text=f"完成 · {spk}人 · {len(result['segments'])}句"))
            self.root.after(0, lambda: self.btn_export.config(state="normal"))
            self.root.after(0, lambda: self.btn_meeting.config(state="normal"))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda: self.lbl_stat.config(text=f"错误: {e}"))
            self.root.after(0, lambda: self.lbl_prog.config(text="出错"))
        self._running = False; self._pipe = None
        self.root.after(0, lambda: self.btn_start.config(state="normal",text="开始"))

    # === 绘制 ===
    def _draw_all(self):
        """绘制波形 + 频率 + 说话人色条"""
        if self._audio is None: return
        w = self.wave.winfo_width(); h = self.wave.winfo_height()
        if w<10: return
        self.wave.delete("all")
        dur = len(self._audio)/self._sr
        show_start = max(0, dur-60); show_end = dur
        s = int(show_start*self._sr); e = int(show_end*self._sr)
        chunk = self._audio[s:e]; cd = len(chunk)/self._sr

        # 波形（绿色）
        step = max(1,len(chunk)//w)
        down = chunk[::step]; coords = []
        for xi in range(min(w,len(down))):
            py = h//2 - int(down[xi]*(h//2-4))
            coords.extend([xi,py])
        if len(coords)>2: self.wave.create_line(*coords,fill="#3fb950",width=1)

        # 音高覆盖（黄色点）
        if self._segments:
            segs_near = [s2 for s2 in self._segments if s2["start"]<show_end and s2["end"]>show_start]
            for seg in segs_near:
                s_t = seg.get("pitch", 80)
                for sec in range(int(seg["start"]), int(seg["end"])+1):
                    if sec < show_start or sec > show_end: continue
                    sx = int((sec-show_start)/cd*w)
                    py = h//2 - int((s_t-50)/300*(h//2-4))
                    self.wave.create_oval(sx-1,py-1,sx+1,py+1,fill="#d29922",outline="")

        # 说话人色条
        by = h-6
        for seg in self._segments:
            sx = int((seg["start"]-show_start)/cd*w) if cd>0 else 0
            ex = int((seg["end"]-show_start)/cd*w) if cd>0 else w
            if sx<0: sx=0; ex=min(ex,w)
            if ex<=sx: continue
            c = SPK_COLORS.get(seg.get("speaker","UNKNOWN"),"#484f58")
            self.wave.create_rectangle(sx,by,ex,h,fill=c,outline="")

        # 时间刻度
        for t in range(int(show_start),int(show_end)+1,10):
            tx = int((t-show_start)/cd*w) if cd>0 else 0
            self.wave.create_text(tx, h-18, text=f"{t//60}:{t%60:02d}",fill="#484f58",font=("Consolas",7),anchor="s")

    def _on_click_wave(self, ev):
        if self._audio is None: return
        w = self.wave.winfo_width(); dur = len(self._audio)/self._sr
        t = max(0,min(dur, dur-60 + (ev.x/w)*60))
        self._play_segment(t,min(t+5,dur))

    # === 时间线（Text widget，轻量）===
    def _add_segments(self, segs):
        """增量追加新句子到 Text widget"""
        self.tl_text.config(state="normal")
        for seg in segs:
            spk = seg.get("speaker","UNKNOWN")
            tag = f"spk{spk[-1]}" if spk.startswith("SPK") else "unknown"
            ts = f"{seg['start']:.1f}s"
            txt = seg["text"][:80] + ("..." if len(seg["text"])>80 else "")

            self.tl_text.insert("end", ts, "ts")
            self.tl_text.insert("end", f" {spk}  ", (tag, "bold"))
            self.tl_text.insert("end", f"{txt}\n", (tag,))

            # 绑定点击事件
            def make_handler(s, e):
                return lambda ev, ss=s, ee=e: self._play_segment(ss, ee)
            self.tl_text.tag_bind(tag, "<Button-1>", make_handler(seg["start"], seg["end"]))

        self.tl_text.see("end")
        self.tl_text.config(state="disabled")

    def _rebuild(self, segs):
        self.tl_text.config(state="normal")
        self.tl_text.delete("1.0", "end")
        self._add_segments(segs)

    # === 播放 ===
    def _play_segment(self, start, end):
        """播放一段音频"""
        if self._audio is None: return
        s = max(0, int(start*self._sr))
        e = min(len(self._audio), int(end*self._sr))
        if e-s < self._sr//10: return

        self.lbl_stat.config(text=f"▶ {start:.1f}s")

        try:
            # 用完整路径（避免短路径 ~ 问题）
            tmp = os.path.join(os.environ['USERPROFILE'], 'AppData', 'Local', 'Temp', 'tp.wav')
            sf.write(tmp, self._audio[s:e], self._sr)
            ctypes.windll.winmm.mciSendStringW('close tp', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW(f'open "{tmp}" alias tp', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW('play tp', None, 0, 0)
        except Exception as ex:
            self.lbl_stat.config(text=f"播放错误: {ex}")

    def _export_report(self):
        """导出JSON报告（含MEITR分析）"""
        if not self._segments:
            self.lbl_stat.config(text="没有可导出的结果")
            return

        # 构建完整报告
        report = {
            "meta": {
                "file": self._current_file or "",
                "duration_sec": round(len(self._audio)/self._sr, 1) if self._audio is not None else 0,
                "total_segments": len(self._segments),
                "num_speakers": len(set(s["speaker"] for s in self._segments)),
            },
            "speakers": {},
            "timeline": [],
            "key_events": [],
            "summary": {},
        }

        spk_stats = {}
        for seg in self._segments:
            spk = seg.get("speaker", "UNKNOWN")
            if spk not in spk_stats:
                spk_stats[spk] = {"count": 0, "total_time": 0.0}
            spk_stats[spk]["count"] += 1
            spk_stats[spk]["total_time"] += seg["end"] - seg["start"]

            # 构建每个片段
            entry = {
                "time": {"start": round(seg["start"], 1), "end": round(seg["end"], 1)},
                "speaker": spk,
                "text": seg["text"],
            }

            # MEITR 分析数据（扁平结构）
            analysis = seg.get("analysis", {})
            if analysis and analysis.get("label"):
                entry["emotion"] = {
                    "label": analysis.get("label", "?"),
                    "3d_vector": {
                        "arousal": analysis.get("arousal", 0),
                        "valence": analysis.get("valence", 0),
                        "dominance": analysis.get("dominance", 0),
                    },
                    "evidence": {
                        "pitch_hz": analysis.get("pitch_hz", 0),
                        "energy": analysis.get("energy", 0),
                        "pitch_std": analysis.get("pitch_std", 0),
                    },
                    "defense": analysis.get("defense_mechanism"),
                    "prediction": analysis.get("prediction", {}),
                }
                entry["conflict"] = {
                    "score": analysis.get("conflict_score", 0),
                    "explanation": analysis.get("conflict_explanation"),
                    "text_analysis": analysis.get("text_sentiment", 0),
                }

                # 关键事件
                exp = analysis.get("conflict_explanation")
                if exp:
                    report["key_events"].append({
                        "time": round(seg["start"], 1),
                        "type": "声文矛盾",
                        "description": exp,
                        "speaker": spk,
                        "conflict_score": analysis.get("conflict_score", 0),
                        "defense": analysis.get("defense_mechanism"),
                    })
                # 爆发警告
                pred = analysis.get("prediction", {})
                if pred.get("alert"):
                    report["key_events"].append({
                        "time": round(seg["start"], 1),
                        "type": "状态预警",
                        "description": pred["alert"],
                        "speaker": spk,
                        "p_suppress": pred.get("p_suppress", 0),
                    })

            report["timeline"].append(entry)

        # 说话人统计
        for spk, st in spk_stats.items():
            report["speakers"][spk] = {
                "sentences": st["count"],
                "total_time_sec": round(st["total_time"], 1),
                "time_percent": round(st["total_time"] / max(report["meta"]["duration_sec"], 1) * 100, 1),
            }

        # 情绪分布统计
        labels = [s.get("analysis", {}).get("label", "?") for s in self._segments if s.get("analysis", {}).get("label")]
        report["summary"]["emotion_distribution"] = {e: labels.count(e) for e in sorted(set(labels))} if labels else {}

        # 矛盾检测汇总
        high_conf = [s for s in self._segments if s.get("analysis", {}).get("conflict_score", 0) > 0.3]
        report["summary"]["conflict_segments"] = len(high_conf)
        report["summary"]["high_risk_segments"] = [
            {"time": round(s["start"], 1), "speaker": s["speaker"], "text": s["text"][:60],
             "conflict_score": s.get("analysis", {}).get("conflict_score", 0)}
            for s in high_conf
        ] if high_conf else []

        # 用户选择保存位置
        from tkinter import filedialog as fd
        out = fd.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON报告", "*.json")],
            initialfile="听心报告_" + time.strftime("%Y%m%d_%H%M%S"))
        if not out:
            return  # 用户取消了

        with open(out, "w", encoding="utf-8") as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)
        os.startfile(out)
        self.lbl_stat.config(text=f"报告已导出: {out}")

        # ── 同时导出文字版报告（技术报告 + 普通人版总结）──
        try:
            from plain_report import generate_technical_report, generate_plain_summary
            txt_path = os.path.splitext(out)[0] + ".txt"
            # 把 segments 转成 plain_report 需要的格式
            data = {
                "filename": os.path.basename(self._current_file) if self._current_file else "",
                "duration": report["meta"]["duration_sec"],
                "num_speakers": report["meta"]["num_speakers"],
                "segments": [],
            }
            for seg in self._segments:
                a = seg.get("analysis", {}) or {}
                data["segments"].append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": seg.get("speaker", "UNKNOWN"),
                    "analysis": {
                        "arousal": a.get("arousal", 0),
                        "valence": a.get("valence", 0),
                        "dominance": a.get("dominance", 0),
                        "label": a.get("label", "?"),
                        "label_conf": a.get("label_conf", 0),
                        "conflict_score": a.get("conflict_score", 0),
                        "conflict_explanation": a.get("conflict_explanation"),
                        "text_sentiment": a.get("text_sentiment", 0),
                        "defense_mechanism": a.get("defense_mechanism"),
                        "defense_confidence": a.get("defense_confidence", 0),
                        "prediction": a.get("prediction", {}),
                        "pitch_hz": a.get("pitch_hz", 0),
                        "energy": a.get("energy", 0),
                        "pitch_std": a.get("pitch_std", 0),
                        "fabrication_score": a.get("fabrication_score", 0),
                        "overcontrol": a.get("overcontrol", 0),
                        "fabrication_evidence": a.get("fabrication_evidence", {}),
                        "context_contradiction": a.get("context_contradiction", 0),
                    }
                })
            tech = generate_technical_report(data)
            plain = generate_plain_summary(data)
            with open(txt_path, "w", encoding="utf-8") as ft:
                ft.write(tech + "\n" + plain)
            self.lbl_stat.config(text=f"JSON + 文字版已导出")
        except ImportError:
            pass  # plain_report 模块不存在时跳过
        except Exception as e:
            self.lbl_stat.config(text=f"报告已导出（文字版生成失败: {e}）")

    
    # ========== 会议纪要 ==========

    def _meeting_direct(self):
        p = filedialog.askopenfilename(filetypes=[("音频","*.mp3 *.wav *.m4a *.flac *.ogg")])
        if not p: return
        self._current_file = p
        self.lbl_stat.config(text=f"处理中: {os.path.basename(p)}")
        self.lbl_prog.config(text="加载...")
        self.btn_meeting.config(state="disabled",text="处理中")
        self._running = True
        threading.Thread(target=self._meeting_run, args=(p,), daemon=True).start()

    def _meeting_run(self, filepath):
        try:
            from core.meeting_minutes import MeetingReport
            audio = load_audio(filepath, sr=self._sr)
            pipe = StreamPipeline(filepath)
            self._pipe = pipe
            def cb(r):
                if r.get("analyzing"):
                    pct = int(r.get("analysis_progress",0)*100)
                    self.root.after(0, lambda p=pct: self.lbl_prog.config(text=f"分析中 {p}%"))
            result = pipe.run(callback=cb, output_dir=str(OUTPUT_DIR))
            segments = result["segments"]
            if not segments:
                self.root.after(0, lambda: self.lbl_stat.config(text="无有效语音段"))
                self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
                return
            mr = MeetingReport()
            spk_stats = {}
            for s in segments:
                sp = s.get("speaker","?")
                if sp not in spk_stats: spk_stats[sp] = {"sentences":0,"time":0.0}
                spk_stats[sp]["sentences"] += 1
                spk_stats[sp]["time"] += s["end"] - s["start"]
            data = mr.generate(segments, spk_stats)
            md = mr.to_markdown(data, filepath)
            out = os.path.splitext(filepath)[0] + "_会议纪要.md"
            with open(out, "w", encoding="utf-8") as f:
                f.write(md)
            self.root.after(0, lambda o=out: os.startfile(o))
            self.root.after(0, lambda: self.lbl_stat.config(text=f"会议纪要: {os.path.basename(out)}"))
            self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda e=e: self.lbl_stat.config(text=f"错误: {e}"))
            self.root.after(0, lambda: self.lbl_prog.config(text="出错"))
        self._running = False; self._pipe = None
        self.root.after(0, lambda: self.btn_meeting.config(state="normal",text="会议纪要"))

    def _batch_meeting_pick(self):
        self._batch_meeting_mode = True
        self._batch_pick()

    def _process_single_batch(self, filepath, idx, total):
        self._current_file = filepath
        audio = load_audio(filepath, sr=self._sr)
        pipe = StreamPipeline(filepath)
        def cb(r):
            if r.get("analyzing"):
                pct = int(r.get("analysis_progress",0)*100)
                self.root.after(0, lambda i=idx,t=total,p=pct:
                    self.lbl_prog.config(text=f"{i}/{t} 分析中 {p}%"))
        result = pipe.run(callback=cb, output_dir=str(OUTPUT_DIR))
        segments = result["segments"]
        basename = os.path.splitext(filepath)[0]
        out = basename + "_听心报告.json"
        report = self._build_report(filepath, audio, segments)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # 保存最后文件数据到实例，供导出报告使用
        self._audio = audio
        self._segments = segments
        if self._batch_meeting_mode and segments:
            from core.meeting_minutes import MeetingReport
            mr = MeetingReport()
            spk_stats = {}
            for s in segments:
                sp = s.get("speaker","?")
                if sp not in spk_stats: spk_stats[sp] = {"sentences":0,"time":0.0}
                spk_stats[sp]["sentences"] += 1
                spk_stats[sp]["time"] += s["end"] - s["start"]
            md_data = mr.generate(segments, spk_stats)
            md = mr.to_markdown(md_data, filepath)
            md_path = basename + "_会议纪要.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)
        self.root.after(0, lambda o=out,fn=os.path.basename(filepath):
            self.lbl_stat.config(text=f"\u2713 {fn} \u2192 {os.path.basename(o)}"))

    # ========== 音乐分析 ==========

    def _music_analyze(self):
        p = filedialog.askopenfilename(filetypes=[("音频","*.mp3 *.wav *.m4a *.flac *.ogg")])
        if not p: return
        self.lbl_stat.config(text=f"分析中: {os.path.basename(p)}")
        self.lbl_prog.config(text="加载...")
        self.btn_music.config(state="disabled",text="分析中")
        threading.Thread(target=self._music_run, args=(p,), daemon=True).start()

    def _music_run(self, filepath):
        try:
            from core.music_analysis import MusicAnalyzer
            from core.preprocess import load_audio
            audio = load_audio(filepath, sr=16000)
            ma = MusicAnalyzer()
            data = ma.analyze(filepath, audio)
            md = ma.to_markdown(data)
            out = os.path.splitext(filepath)[0] + "_音乐分析报告.md"
            with open(out, "w", encoding="utf-8") as f:
                f.write(md)
            self.root.after(0, lambda o=out: os.startfile(o))
            self.root.after(0, lambda: self.lbl_stat.config(text=f"音乐分析: {os.path.basename(out)}"))
            self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda e=e: self.lbl_stat.config(text=f"错误: {e}"))
            self.root.after(0, lambda: self.lbl_prog.config(text="出错"))
        self.root.after(0, lambda: self.btn_music.config(state="normal",text="音乐分析"))

    def _batch_music_pick(self):
        d = filedialog.askdirectory()
        if not d: return
        exts = ('.mp3','.wav','.m4a','.flac','.ogg','.aac')
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts))
        if not files: self.lbl_stat.config(text="无音乐文件"); return
        self.lbl_stat.config(text=f"批量音乐 {len(files)}个")
        self._running = True
        self.btn_batch_music.config(state="disabled",text="批量中...")
        threading.Thread(target=self._batch_music_proc, args=(files,), daemon=True).start()

    def _batch_music_proc(self, files):
        from core.music_analysis import MusicAnalyzer
        from core.preprocess import load_audio
        ma = MusicAnalyzer()
        ok, fail = 0, 0
        for i, fp in enumerate(files):
            fname = os.path.basename(fp)
            self.root.after(0, lambda i=i+1,t=len(files),n=fname:
                self.lbl_stat.config(text=f"音乐 {i}/{t}: {n}"))
            try:
                audio = load_audio(fp, sr=16000)
                data = ma.analyze(fp, audio)
                md = ma.to_markdown(data)
                out_path = os.path.splitext(fp)[0] + "_音乐分析报告.md"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(md)
                ok += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                fail += 1
        self.root.after(0, lambda: self.lbl_stat.config(text=f"音乐批量: {ok}成功/{fail}失败"))
        self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
        self._running = False
        self.root.after(0, lambda: self.btn_batch_music.config(state="normal",text="批量音乐"))

    # ========== 通话分析 ==========

    def _call_analyze(self):
        p = filedialog.askopenfilename(filetypes=[("JSON报告","*.json")])
        if not p: return
        self.lbl_stat.config(text=f"分析: {os.path.basename(p)}")
        try:
            with open(p, 'r', encoding='utf-8') as f:
                report = json.load(f)
            from core.call_analysis import CallAnalyzer, CallReport
            segments = report.get("timeline", [])
            if not segments:
                self.lbl_stat.config(text="报告无时间线数据"); return
            ca = CallAnalyzer()
            data = ca.analyze(segments)
            md = CallReport().to_markdown(single=data, filename=p)
            out = os.path.splitext(p)[0] + "_通话分析.md"
            with open(out, 'w', encoding='utf-8') as f:
                f.write(md)
            os.startfile(out)
            self.lbl_stat.config(text=f"通话分析: {os.path.basename(out)}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_stat.config(text=f"错误: {e}")

    def _batch_call_pick(self):
        d = filedialog.askdirectory()
        if not d: return
        jsons = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json") and "\u62a5\u544a" in f])
        if not jsons: self.lbl_stat.config(text="文件夹无听心报告JSON"); return
        self.lbl_stat.config(text=f"话务 {len(jsons)}通分析中...")
        threading.Thread(target=self._batch_call_proc, args=(jsons, d), daemon=True).start()

    def _batch_call_proc(self, jsons, out_dir):
        try:
            from core.call_analysis import BatchCallAnalyzer, CallReport
            reports = []
            for jp in jsons:
                with open(jp, 'r', encoding='utf-8') as f:
                    reports.append(json.load(f))
            bca = BatchCallAnalyzer()
            data = bca.analyze(reports)
            md = CallReport().to_markdown(batch=data)
            out = os.path.join(out_dir, "\u8bdd\u52a1\u6279\u91cf\u5206\u6790\u62a5\u544a.md")
            with open(out, 'w', encoding='utf-8') as f:
                f.write(md)
            self.root.after(0, lambda o=out: os.startfile(o))
            self.root.after(0, lambda: self.lbl_stat.config(text="话务报告: 话务批量分析报告.md"))
            self.root.after(0, lambda: self.lbl_prog.config(text="完成"))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.root.after(0, lambda e=e: self.lbl_stat.config(text=f"错误: {e}"))

    # ========== 报告构建（复用） ==========

    def _build_report(self, filepath, audio, segments):
        report = {
            "meta": {
                "file": filepath or "",
                "duration_sec": round(len(audio)/self._sr, 1) if audio is not None else 0,
                "total_segments": len(segments),
                "num_speakers": len(set(s["speaker"] for s in segments)),
            },
            "speakers": {},
            "timeline": [],
            "key_events": [],
            "summary": {},
        }
        spk_stats = {}
        for seg in segments:
            spk = seg.get("speaker", "UNKNOWN")
            if spk not in spk_stats:
                spk_stats[spk] = {"count": 0, "total_time": 0.0}
            spk_stats[spk]["count"] += 1
            spk_stats[spk]["total_time"] += seg["end"] - seg["start"]
            entry = {
                "time": {"start": round(seg["start"], 1), "end": round(seg["end"], 1)},
                "speaker": spk,
                "text": seg["text"],
            }
            analysis = seg.get("analysis", {})
            if analysis and analysis.get("label"):
                entry["emotion"] = {
                    "label": analysis.get("label", "?"),
                    "3d_vector": {
                        "arousal": analysis.get("arousal", 0),
                        "valence": analysis.get("valence", 0),
                        "dominance": analysis.get("dominance", 0),
                    },
                    "evidence": {
                        "pitch_hz": analysis.get("pitch_hz", 0),
                        "energy": analysis.get("energy", 0),
                        "pitch_std": analysis.get("pitch_std", 0),
                    },
                    "defense": analysis.get("defense_mechanism"),
                    "prediction": analysis.get("prediction", {}),
                }
                entry["conflict"] = {
                    "score": analysis.get("conflict_score", 0),
                    "explanation": analysis.get("conflict_explanation"),
                    "text_analysis": analysis.get("text_sentiment", 0),
                }
                exp = analysis.get("conflict_explanation")
                if exp:
                    report["key_events"].append({
                        "time": round(seg["start"], 1), "type": "\u58f0\u6587\u77db\u76fe",
                        "description": exp, "speaker": spk,
                        "conflict_score": analysis.get("conflict_score", 0),
                        "defense": analysis.get("defense_mechanism"),
                    })
                pred = analysis.get("prediction", {})
                if pred.get("alert"):
                    report["key_events"].append({
                        "time": round(seg["start"], 1), "type": "\u72b6\u6001\u9884\u8b66",
                        "description": pred["alert"], "speaker": spk,
                        "p_suppress": pred.get("p_suppress", 0),
                    })
            report["timeline"].append(entry)
        for spk, st in spk_stats.items():
            report["speakers"][spk] = {
                "sentences": st["count"],
                "total_time_sec": round(st["total_time"], 1),
                "time_percent": round(st["total_time"] / max(report["meta"]["duration_sec"], 1) * 100, 1),
            }
        labels = [s.get("analysis", {}).get("label", "?") for s in segments if s.get("analysis", {}).get("label")]
        report["summary"]["emotion_distribution"] = {e: labels.count(e) for e in sorted(set(labels))} if labels else {}
        high_conf = [s for s in segments if s.get("analysis", {}).get("conflict_score", 0) > 0.3]
        report["summary"]["conflict_segments"] = len(high_conf)
        report["summary"]["high_risk_segments"] = [
            {"time": round(s["start"], 1), "speaker": s["speaker"], "text": s["text"][:60],
             "conflict_score": s.get("analysis", {}).get("conflict_score", 0)}
            for s in high_conf
        ] if high_conf else []
        return report

    def _on_close(self):
        if self._pipe: self._pipe.abort()
        self.root.destroy()

if __name__=="__main__":
    app = TingXin()
    app.root.mainloop()
