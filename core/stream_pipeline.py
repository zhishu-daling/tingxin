"""
听心 · 流式管线（60秒块 + whisper small CPU）

60秒一大块 → whisper批处理 → VAD分句 → 逐句显示
比VAD逐句跑快5倍，比全量跑只慢一点但能流式出
"""
import json, os, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .asr import transcribe
from .preprocess import load_audio, split_audio, SAMPLE_RATE
from .diarize import IncrementalDiarizer, diarize
from .acoustic_features import extract_segment_features
from .analysis import (compute_arousal, compute_valence, compute_dominance,
                     emotion_label, detect_defense, detect_conflict,
                     InteractionAnalyzer, StatePredictor,
                     SpeakerBaseline, compute_context_contradiction,
                     compute_fabrication_score)


class StreamPipeline:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self._aborted = False

    def abort(self):
        self._aborted = True

    def _vad_segment(self, audio, sr=SAMPLE_RATE):
        """能量检测切句子（用于分句显示）"""
        frame_ms = 30; frame_len = sr * frame_ms // 1000
        silence_ms = 400; silence_len = sr * silence_ms // 1000
        thresh = 0.015

        segs = []; in_speech = False; speech_start = 0; silence = 0
        for i in range(0, len(audio) - frame_len + 1, frame_len):
            energy = np.sqrt(np.mean(audio[i:i+frame_len]**2))
            if energy > thresh:
                if not in_speech: in_speech = True; speech_start = i; silence = 0
                else: silence = 0
            elif in_speech:
                silence += frame_len
                if silence >= silence_len:
                    segs.append((speech_start, i - silence + frame_len)); in_speech = False; silence = 0
        if in_speech: segs.append((speech_start, len(audio)))
        return [(s,e) for s,e in segs if e-s >= sr//2]

    def run(self, language="zh", use_diarization=True, callback=None, output_dir=None):
        start_time = time.time()
        audio = load_audio(self.audio_path, sr=SAMPLE_RATE)
        total_samples = len(audio)

        # VAD 切句（全量，用于显示）
        all_vad = self._vad_segment(audio)
        print(f"[Stream] VAD: {len(all_vad)} 句")

        # 预加载 + GPU预热
        print("[Stream] 预热 GPU...")
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        transcribe(dummy, SAMPLE_RATE)
        print("[Stream] GPU 就绪")

        # 60秒一大块处理
        chunks = split_audio(audio, SAMPLE_RATE, segment_ms=60000)
        print(f"[Stream] {len(chunks)} 块 (60s/块)")

        all_segments = []
        diarizer = IncrementalDiarizer(n_speakers=2) if use_diarization else None
        if diarizer:
            diarizer.set_audio(audio)

        for ci, (cs, ce, chunk) in enumerate(chunks):
            if self._aborted: break

            # ASR 批处理一整块（持久子进程）
            t0 = time.time()
            asr_segs = transcribe(chunk, SAMPLE_RATE)
            dt = time.time() - t0

            # 调整时间偏移，先标临时说话人
            chunk_segments = []
            for seg in asr_segs:
                seg["start"] = round(seg["start"] + cs / SAMPLE_RATE, 2)
                seg["end"] = round(seg["end"] + cs / SAMPLE_RATE, 2)
                seg["speaker"] = f"TMP{len(all_segments)+1}"
                chunk_segments.append(seg)
                all_segments.append(seg)

            # 说话人聚类（每块结束后对全部积累片段做一次K-Means）
            if diarizer and len(all_segments) >= 2:
                all_segments = diarizer.refine(all_segments)

            progress = (ci + 1) / len(chunks)
            if callback:
                callback({
                    "progress": progress,
                    "new_segments": chunk_segments,
                    "total_segments": len(all_segments),
                })

        # 最终精校
        final = all_segments
        if diarizer and len(final) > 1:
            print(f"[Stream] 精校 {len(final)} 段...")
            refined = diarizer.batch_refine(final, self.audio_path)
            if callback: callback({"progress": 1.0, "refined": refined, "total_segments": len(refined)})
            final = refined

        # 感知 + 伪装分析（两阶段：先建立基线，再计算伪装分数）
        print(f"[Stream] 分析 {len(final)} 段...")
        interaction = InteractionAnalyzer()
        predictor = StatePredictor()

        # 阶段一：所有段的声学特征 + 基线累积（多线程并行）
        print(f"[Stream] 提取 {len(final)} 段特征...")
        
        # 1. 准备所有段的数据
        seg_tasks = []
        for seg in final:
            s = int(seg["start"] * SAMPLE_RATE)
            e = int(seg["end"] * SAMPLE_RATE)
            s = max(0,s); e = min(len(audio),e)
            chunk = audio[s:e] if e>s else None
            seg_tasks.append((seg, chunk))

        # 2. 多线程并行跑 extract_segment_features
        total_tasks = len(seg_tasks)
        completed = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_to_seg = {executor.submit(extract_segment_features, chunk, SAMPLE_RATE): seg
                         for seg, chunk in seg_tasks if chunk is not None}
            # 空chunk直接赋空
            for seg, chunk in seg_tasks:
                if chunk is None:
                    seg["_feat"] = {}

            for fut in as_completed(fut_to_seg):
                seg = fut_to_seg[fut]
                try:
                    seg["_feat"] = fut.result()
                except Exception as e:
                    print(f"[Stream] 特征提取异常: {e}")
                    seg["_feat"] = {}
                completed += 1
                # 每完成10%报一次进度
                if callback and completed % max(1, total_tasks // 10) == 0:
                    callback({"progress": 1.0, "analyzing": True, "analysis_progress": completed / total_tasks})

        # 3. 串行更新基线
        baselines = {}
        for seg in final:
            spk = seg.get("speaker", "UNKNOWN")
            if spk not in baselines:
                baselines[spk] = SpeakerBaseline()
            feat = seg.get("_feat", {})
            dur = seg["end"] - seg["start"]
            baselines[spk].update(feat, dur)

        # 阶段二：逐段分析（此时基线已建立）
        total = len(final)
        for i, seg in enumerate(final):
            spk = seg.get("speaker", "UNKNOWN")
            feat = seg.get("_feat", {})
            baseline = baselines.get(spk, SpeakerBaseline())

            a_val, a_ev = compute_arousal(feat)
            v_val, v_ev = compute_valence(feat)
            d_val, d_ev = compute_dominance(feat)
            label, conf = emotion_label(a_val, v_val, d_val)
            text = seg["text"]
            c_score, c_exp, c_st = detect_conflict(text, v_val, d_val, a_val)
            mech, mech_conf = detect_defense({"arousal":a_val,"valence":v_val,"dominance":d_val,"conflict_score":c_score}, text)
            pred = predictor.update(a_val, v_val, d_val, c_score)
            interaction.add_segment(seg, feat)

            # 共谋伪装检测
            ctx_con = compute_context_contradiction(i, final)
            fab_score, overcontrol, fab_ev = compute_fabrication_score(feat, baseline, ctx_con)

            seg["analysis"] = {
                "arousal": round(a_val,3), "valence": round(v_val,3), "dominance": round(d_val,3),
                "label": label, "label_conf": conf,
                "arousal_evidence": a_ev, "valence_evidence": v_ev, "dominance_evidence": d_ev,
                "conflict_score": c_score, "conflict_explanation": c_exp,
                "text_sentiment": c_st,
                "defense_mechanism": mech, "defense_confidence": mech_conf,
                "prediction": pred,
                "pitch_hz": round(feat.get("pitch_mean",0),1) if feat else 0,
                "energy": round(feat.get("energy_mean",0),4) if feat else 0,
                "pitch_std": round(feat.get("pitch_std",0),1) if feat else 0,
                "fabrication_score": fab_score,
                "overcontrol": overcontrol,
                "fabrication_evidence": fab_ev,
                "context_contradiction": ctx_con,
            }

            # 清理临时字段
            seg.pop("_feat", None)

            # 每5段推进度（避免GUI过载）
            if callback and i % 5 == 0:
                callback({"progress": 1.0, "analyzing": True, "analysis_progress": (i+1) / total})

        t = time.time() - start_time
        result = {
            "file": self.audio_path, "filename": os.path.basename(self.audio_path),
            "duration": round(total_samples / SAMPLE_RATE, 2), "total_time": round(t, 2),
            "num_speakers": len(set(s["speaker"] for s in final)), "segments": final,
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, os.path.splitext(os.path.basename(self.audio_path))[0] + ".json")
            with open(out, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
        return result
