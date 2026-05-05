"""
Dyno RDG — Tkinter desktop UI
Self-contained: no backend server required.
Data model: Transcript is the single source of truth.
Event rows are managed directly inside App.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx                         # noqa: F401
import numpy                            # noqa: F401
import pandas                           # noqa: F401
from PIL import Image, ImageTk

from Bio import Entrez, SeqIO

from dynordg import Transcript
from dynordg import RiboGraphFlux
from dynordg import RiboGraphVis

from collections import defaultdict

Entrez.email = "user@example.com"

# ── Colours ───────────────────────────────────────────────────────────────────
BG      = "#0e0f11"
SURFACE = "#16181c"
BORDER  = "#2a2d35"
ACCENT  = "#00e5a0"
ACCENT2 = "#0077ff"
DANGER  = "#ff4455"
WARN    = "#ffaa00"
TEXT    = "#e2e4ea"
MUTED   = "#5a5f72"
AUTO_FG = "#7ecfff"

STYLE = {
    "font_mono":       ("Courier", 11),
    "font_mono_small": ("Courier", 9),
    "font_header":     ("Courier", 12, "bold"),
    "sidebar_width":   280,
    "window_width":    1340,
    "window_height":   860,
}
MONO   = STYLE["font_mono"]
MONO_S = STYLE["font_mono_small"]

MAX_EVENTS  = 1000
EVENT_TYPES = ["initiation", "termination", "ires", "frameshift+1", "frameshift-1", "cap", 'load_scanning', 'retention']

# ── Per-event probability constraints ─────────────────────────────────────────
# prob_max
EVENT_CONSTRAINTS: dict[str, float] = {
    "cap":         1.0,
    "ires":        1.0,
    "initiation":  1.0,
    "termination": 1.0,
    "frameshift+1":     1.0,
    "frameshift-1":     1.0,
}

def _event_constraints(etype: str) -> float:
    return EVENT_CONSTRAINTS.get(etype, None)

# ── RiboGraphFlux parameters ──────────────────────────────────────────────────
FLUX_PARAM_SCHEMA = {
    "half_life_scanning":       {"type": "float", "default": None,  "min": 0.0, "max": 10000.0, "nullable": True},
    "half_life_translation":    {"type": "float", "default": None,  "min": 0.0, "max": 10000.0, "nullable": True},
    "weight_cutoff":            {"type": "float", "default": 0.0,   "min": 0.0, "max": 1.0,     "nullable": False},
    "reinitiation_half_life":   {"type": "float", "default": None,  "min": 0.0, "max": 10000.0, "nullable": True},
    "ternary_complex_half_life":{"type": "float", "default": None,  "min": 0.0, "max": 10000.0, "nullable": True},
    "flux_cutoff":              {"type": "float", "default": 0.001, "min": 0.0, "max": 1.0,     "nullable": False},
    "retention_limit":          {"type": "int",   "default": 1,     "min": 0,   "max": 10,      "nullable": True},
}
FLUX_PARAM_LABELS = {
    "half_life_scanning":       "Scanning half-life (nt)",
    "half_life_translation":    "Translation half-life (nt)",
    "weight_cutoff":            "Weight cutoff",
    "reinitiation_half_life":   "Reinitiation half-life (nt)",
    "ternary_complex_half_life":"Ternary complex half-life (nt)",
    "flux_cutoff":              "Flux cutoff",
    "retention_limit":          "Retention limit",
}
VIS_PARAM_SCHEMA  = {"log_reduction": {"type": "float", "default": 1.5, "min": 1.0, "max": 10.0, "nullable": False}}
VIS_PARAM_LABELS  = {"log_reduction": "Log reduction"}
PARAM_SCHEMA      = {**FLUX_PARAM_SCHEMA, **VIS_PARAM_SCHEMA}
PARAM_LABELS      = {**FLUX_PARAM_LABELS, **VIS_PARAM_LABELS}

# ── Colour helpers ────────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def blend(fg, bg, alpha):
    fr, fg_, fb = hex_to_rgb(fg)
    br, bg_, bb = hex_to_rgb(bg)
    return rgb_to_hex(int(br+(fr-br)*alpha), int(bg_+(fg_-bg_)*alpha), int(bb+(fb-bb)*alpha))

# ── Input validators ──────────────────────────────────────────────────────────
def _int_validator(widget, allow_negative=False):
    def ok(P):
        if P == "": return True
        if allow_negative and P == "-": return True
        try: int(P); return True
        except: return False
    return (widget.register(ok), "%P")

def _float_validator(widget, allow_negative=False):
    def ok(P):
        if P in ("", "."): return True
        if allow_negative and P in ("-", "-."): return True
        try: float(P); return True
        except: return False
    return (widget.register(ok), "%P")

# ── Styled primitives ─────────────────────────────────────────────────────────
def styled_entry(parent, width=8, **kw):
    return tk.Entry(parent, width=width, bg=BG, fg=TEXT,
                    insertbackground=ACCENT, relief="flat",
                    highlightthickness=1, highlightbackground=BORDER,
                    highlightcolor=ACCENT, font=MONO_S, **kw)

def styled_button(parent, text, command, style="primary", **kw):
    colours = {"primary": (ACCENT,"#000"), "secondary": (SURFACE,ACCENT2),
               "danger": (SURFACE,DANGER), "warn": (SURFACE,WARN)}
    bg, fg = colours.get(style, (ACCENT,"#000"))
    btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                    activebackground=blend(bg,"#ffffff",0.15), activeforeground=fg,
                    relief="flat", font=MONO_S, cursor="hand2", padx=10, pady=4, **kw)
    if style in ("secondary","danger","warn"):
        btn.config(highlightthickness=1,
                   highlightbackground={"secondary":ACCENT2,"danger":DANGER,"warn":WARN}[style])
    return btn

def label(parent, text, muted=False, **kw):
    return tk.Label(parent, text=text, bg=kw.pop("bg",SURFACE),
                    fg=MUTED if muted else TEXT, font=MONO_S, **kw)

def section_label(parent, text, bg=BG):
    return tk.Label(parent, text=text, bg=bg, fg=MUTED, font=MONO_S, anchor="w")

# ─────────────────────────────────────────────────────────────────────────────
# SVGCanvas
# ─────────────────────────────────────────────────────────────────────────────
class SVGCanvas(tk.Canvas):
    _SPINNER = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, parent, on_svg_ready=None, **kw):
        super().__init__(parent, bg=SURFACE, highlightthickness=0, **kw)
        self._png_bytes = None
        self._photo     = None
        self._svg_text  = None
        self._on_svg_ready = on_svg_ready
        self._loading_job  = None
        self.bind("<Configure>", self._on_resize)

    def show_loading(self):
        self._stop_spinner()
        self._png_bytes = None; self._photo = None
        self.delete("all")
        self._spinner_idx = 0
        self._spinner_id  = self.create_text(
            max(self.winfo_width()//2, 400), max(self.winfo_height()//2, 200),
            text=f"{self._SPINNER[0]}  Rendering…", fill=MUTED, font=MONO)
        self._tick_spinner()

    def _tick_spinner(self):
        self._spinner_idx = (self._spinner_idx+1) % len(self._SPINNER)
        self.itemconfigure(self._spinner_id,
                           text=f"{self._SPINNER[self._spinner_idx]}  Rendering…")
        self._loading_job = self.after(80, self._tick_spinner)

    def _stop_spinner(self):
        if self._loading_job:
            self.after_cancel(self._loading_job)
            self._loading_job = None

    def load_svg(self, svg_text, png_bytes=None):
        self._stop_spinner()
        self._svg_text  = svg_text
        self._png_bytes = png_bytes
        if self._on_svg_ready: self._on_svg_ready(svg_text)
        self._draw()

    def _on_resize(self, _e):
        if self._png_bytes: self._draw()

    def _draw(self):
        if not self._png_bytes: return
        try:
            cw = max(self.winfo_width(), 100)
            ch = max(self.winfo_height(), 100)
            img = Image.open(io.BytesIO(self._png_bytes))
            img.thumbnail((cw, ch), Image.LANCZOS)
            self.delete("all")
            photo = ImageTk.PhotoImage(img)
            self._photo = photo
            self.create_image((cw-img.width)//2, (ch-img.height)//2,
                              anchor="nw", image=photo)
        except Exception as e:
            self.delete("all")
            self.create_text(self.winfo_width()//2 or 400,
                             self.winfo_height()//2 or 200,
                             text=f"Display error:\n{e}", fill=DANGER, font=MONO_S)

    def clear(self, message="No transcript loaded — add events or load a sequence"):
        self._stop_spinner()
        self._png_bytes = None; self._photo = None
        self.delete("all")
        self.create_text(max(self.winfo_width()//2, 400),
                         max(self.winfo_height()//2, 200),
                         text=message, fill=MUTED, font=MONO)

# ─────────────────────────────────────────────────────────────────────────────
# SequencePanel
# ─────────────────────────────────────────────────────────────────────────────
class SequencePanel(tk.Frame):
    def __init__(self, parent, on_transcript, on_dirty=None, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.on_transcript = on_transcript
        self.on_dirty      = on_dirty or (lambda: None)
        self._transcript: Optional[Transcript] = None
        self._build()

    def _build(self):
        acc_frame = tk.Frame(self, bg=SURFACE)
        acc_frame.pack(fill="x", padx=8, pady=(8,2))
        label(acc_frame, "Accession:", bg=SURFACE).pack(side="left")
        self._acc_var = tk.StringVar()
        acc_e = styled_entry(acc_frame, width=14, textvariable=self._acc_var)
        acc_e.pack(side="left", padx=(4,6))
        acc_e.bind("<Return>", lambda _e: self._fetch_accession())
        styled_button(acc_frame, "Fetch", self._fetch_accession, style="secondary").pack(side="left")

        seq_frame = tk.Frame(self, bg=SURFACE)
        seq_frame.pack(fill="x", padx=8, pady=(6,2))
        label(seq_frame, "Sequence (paste or load FASTA):", bg=SURFACE).pack(anchor="w")
        text_scroll = ttk.Scrollbar(seq_frame, orient="vertical")
        self._seq_text = tk.Text(seq_frame, height=5, width=28, bg=BG, fg=TEXT,
                                  insertbackground=ACCENT, relief="flat",
                                  font=("Courier",9), wrap="char",
                                  highlightthickness=1, highlightbackground=BORDER,
                                  highlightcolor=ACCENT, yscrollcommand=text_scroll.set)
        text_scroll.config(command=self._seq_text.yview)
        text_scroll.pack(side="right", fill="y")
        self._seq_text.pack(fill="x")
        self._seq_text.bind("<KeyRelease>", lambda _e: self.on_dirty())

        btn_row = tk.Frame(self, bg=SURFACE)
        btn_row.pack(fill="x", padx=8, pady=(4,0))
        styled_button(btn_row, "↑ Load FASTA", self._load_fasta, style="secondary").pack(side="left", padx=(0,4))
        styled_button(btn_row, "Apply Sequence", self._apply_sequence, style="primary").pack(side="left")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=8, pady=8)
        label(self, "AUTO-DETECT PARAMETERS", muted=True, bg=SURFACE).pack(anchor="w", padx=8, pady=(0,4))

        self._cutoff_var = tk.DoubleVar(value=0.0)
        self._reinit_var = tk.DoubleVar(value=0.0)
        self._make_float_row("Init score cutoff:", self._cutoff_var, 0.0, 1.0)
        self._make_float_row("Reinitiation prob:", self._reinit_var, 0.0, 1.0)

        styled_button(self, "⟳ Re-run Auto-detect", self._rerun_autodetect,
                      style="warn").pack(fill="x", padx=8, pady=(6,4))

        self._status_var = tk.StringVar(value="No sequence loaded.")
        tk.Label(self, textvariable=self._status_var, bg=SURFACE, fg=MUTED,
                 font=MONO_S, anchor="w", wraplength=240).pack(fill="x", padx=8, pady=(0,8))

    def _make_float_row(self, lbl, var, lo, hi):
        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill="x", padx=8, pady=2)
        label(row, lbl, bg=SURFACE).pack(side="left")
        entry = styled_entry(row, width=7, textvariable=var)
        entry.config(validate="key", validatecommand=_float_validator(entry))
        def commit(_e):
            try: var.set(round(max(lo, min(hi, float(entry.get()))), 10))
            except: var.set(0.0)
        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.pack(side="left", padx=(4,0))

    def _fetch_accession(self):
        acc = self._acc_var.get().strip()
        if not acc:
            messagebox.showwarning("No accession", "Enter an accession number first.")
            return
        self._status_var.set(f"Fetching {acc}…")
        self.update_idletasks()
        def do_fetch():
            try:
                handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
                record = SeqIO.read(handle, "fasta")
                handle.close()
                self.after(0, lambda s=str(record.seq), r=record:
                           self._load_sequence_string(s, id=r.id, description=r.description))
            except Exception as e:
                self.after(0, lambda m=str(e): (
                    self._status_var.set(f"Fetch failed: {m}"),
                    messagebox.showerror("Fetch failed", m)))
        threading.Thread(target=do_fetch, daemon=True).start()

    def _load_fasta(self):
        path = filedialog.askopenfilename(
            filetypes=[("FASTA files","*.fa *.fasta *.fna *.txt"),("All files","*.*")])
        if not path: return
        try:
            record = SeqIO.read(path, "fasta")
            self._load_sequence_string(str(record.seq), id=record.id,
                                       description=record.description)
        except Exception as e:
            messagebox.showerror("FASTA error", str(e))

    def _load_sequence_string(self, seq, id="<unknown id>", description=""):
        self._seq_text.delete("1.0","end")
        self._seq_text.insert("1.0", seq)
        self._apply_sequence(id=id, description=description)

    def _apply_sequence(self, *_args, id="<unknown id>", description=""):
        raw   = self._seq_text.get("1.0","end").strip()
        lines = [l for l in raw.splitlines() if not l.startswith(">")]
        seq   = "".join(lines).replace(" ","").upper()
        if not seq:
            messagebox.showwarning("Empty sequence","Paste or load a sequence first.")
            return
        try:
            t = Transcript(seq, id=id, description=description)
            t.auto_stop_starts(cutoff=self._cutoff_var.get(),
                               reinitiation_prob=self._reinit_var.get())
            self._transcript = t
            self._status_var.set(f"{id}  |  {len(t)} nt  |  "
                                 f"{sum(len(v) for v in t.events.values())} auto-events")
            self.on_transcript(t)
        except ValueError as e:
            messagebox.showerror("Sequence error", str(e))

    def _rerun_autodetect(self):
        if self._transcript is None:
            messagebox.showwarning("No transcript","Apply a sequence first.")
            return
        t     = self._transcript
        fresh = Transcript(str(t.seq), id=t.id, description=t.description)
        fresh.auto_stop_starts(cutoff=self._cutoff_var.get(),
                               reinitiation_prob=self._reinit_var.get())
        for pos, events in t.events.items():
            for etype, edata in events.items():
                if edata.get("_manual"):
                    fresh.events[pos][etype] = edata
        self._transcript = fresh
        self._status_var.set(f"{t.id}  |  {len(fresh)} nt  |  "
                             f"{sum(len(v) for v in fresh.events.values())} events after re-detect")
        self.on_transcript(fresh)

    @property
    def transcript(self): return self._transcript

# ─────────────────────────────────────────────────────────────────────────────
# ParamPanel
# ─────────────────────────────────────────────────────────────────────────────
class ParamPanel(tk.Frame):
    def __init__(self, parent, on_change, on_dirty=None, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.on_change = on_change
        self.on_dirty  = on_dirty or (lambda: None)
        self._vars:    dict[str, tk.StringVar] = {}
        self._entries: dict[str, tk.Entry]     = {}
        self._build()

    def _build(self):
        for section_name, schema, labels in (
            ("FLUX", FLUX_PARAM_SCHEMA, FLUX_PARAM_LABELS),
            ("VIS",  VIS_PARAM_SCHEMA,  VIS_PARAM_LABELS),
        ):
            tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=4, pady=(8,2))
            tk.Label(self, text=section_name, bg=SURFACE, fg=MUTED,
                     font=MONO_S, anchor="w").pack(fill="x", padx=4)
            for k, spec in schema.items():
                tk.Label(self, text=labels[k]+":", bg=SURFACE, fg=TEXT,
                         font=MONO_S, anchor="w").pack(fill="x", padx=4, pady=(4,0))
                display = "" if spec["default"] is None else str(spec["default"])
                var     = tk.StringVar(value=display)
                entry   = styled_entry(self, width=10, textvariable=var)
                vtype   = spec["type"]
                entry.config(validate="key",
                             validatecommand=(_int_validator(entry) if vtype=="int"
                                              else _float_validator(entry)))
                nullable = spec.get("nullable", False)
                if nullable and display == "":
                    entry.insert(0, "None"); entry.config(fg=MUTED)
                def _fin(e, en=entry): en.config(fg=MUTED if en.get()=="" else TEXT)
                def _fout(e, en=entry, k=k, v=var, s=spec):
                    if en.get().strip() == "":
                        en.delete(0,"end"); en.insert(0,"None"); en.config(fg=MUTED)
                    self._commit(k, v, s, en)
                if nullable:
                    entry.bind("<FocusIn>",  lambda e, en=entry: (
                        en.delete(0,"end") if en.get()=="None" else None,
                        en.config(fg=TEXT)))
                    entry.bind("<FocusOut>", _fout)
                    entry.bind("<Return>",   lambda e, k=k, v=var, s=spec, en=entry:
                                             self._commit(k, v, s, en))
                else:
                    entry.bind("<Return>",   lambda e, k=k, v=var, s=spec, en=entry:
                                             self._commit(k, v, s, en))
                    entry.bind("<FocusOut>", lambda e, k=k, v=var, s=spec, en=entry:
                                             self._commit(k, v, s, en))
                var.trace_add("write", lambda *_: self.on_dirty())
                entry.pack(anchor="w", padx=4, pady=(1,0))
                self._vars[k]    = var
                self._entries[k] = entry

    def _commit(self, key, var, spec, entry):
        raw = entry.get().strip()
        if raw in ("","None"):
            if not spec.get("nullable"): var.set(str(spec["default"]))
            self.on_change(); return
        try:
            val = int(raw) if spec["type"]=="int" else float(raw)
            val = max(spec["min"], min(spec["max"], val))
            var.set(str(val))
        except (ValueError, tk.TclError):
            var.set("" if spec.get("nullable") else str(spec["default"]))
        self.on_change()

    def get_values(self):
        out = {}
        for k, var in self._vars.items():
            raw = var.get().strip()
            if raw in ("","None"):
                out[k] = None
            else:
                try:
                    out[k] = int(raw) if PARAM_SCHEMA[k]["type"]=="int" else float(raw)
                except (ValueError, tk.TclError):
                    out[k] = PARAM_SCHEMA[k]["default"]
        return out

    def set_values(self, values):
        for k, var in self._vars.items():
            if k not in values: continue
            val   = values[k]
            entry = self._entries.get(k)
            if val is None:
                var.set(""); entry and (entry.delete(0,"end"),
                                        entry.insert(0,"None"),
                                        entry.config(fg=MUTED))
            else:
                var.set(str(val)); entry and entry.config(fg=TEXT)

# ─────────────────────────────────────────────────────────────────────────────
# Render bridge
# ─────────────────────────────────────────────────────────────────────────────
def render_transcript(transcript, params):
    try:
        tm   = transcript.transition_map()
        flux = RiboGraphFlux(
            transition_map=tm,
            half_life_scanning=        params.get("half_life_scanning"),
            half_life_translation=     params.get("half_life_translation"),
            weight_cutoff=             params.get("weight_cutoff")  or 0.0,
            reinitiation_half_life=    params.get("reinitiation_half_life"),
            ternary_complex_half_life= params.get("ternary_complex_half_life"),
            flux_cutoff=               params.get("flux_cutoff")    or 0.001,
            retention_limit=           params.get("retention_limit"),
        )
        vis = RiboGraphVis(incoming_graph_data=flux,
                           log_scale=float(params.get("log_reduction") or 1.5))
        png_buf = io.BytesIO()
        vis.fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
        svg_buf = io.BytesIO()
        vis.fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        plt.close(vis.fig)
        return svg_buf.getvalue().decode("utf-8"), png_buf.getvalue()
    except Exception:
        return traceback.format_exc(), None

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    # Column definitions for the event table
    _ET_COLS   = ["pos", "type", "prob"]
    _ET_LABELS = ["Position", "Event Type", "Probability"]

    def __init__(self):
        super().__init__()
        self.title("Dyno RDG")
        self.geometry(f"{STYLE['window_width']}x{STYLE['window_height']}")
        self.configure(bg=BG)
        self.minsize(1100, 640)

        self._render_timer = None
        self._last_svg:   str | None = None
        self._render_gen: int        = 0
        self._transcript: Optional[Transcript] = None
        self._row_widgets: list[list[tk.Widget]] = []  # one entry per visible row

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=SURFACE, height=48)
        header.pack(fill="x", side="top"); header.pack_propagate(False)
        tk.Label(header, text="DYNO  RDG", bg=SURFACE, fg=ACCENT,
                 font=STYLE["font_header"]).pack(side="left", padx=20)
        tk.Label(header, text="v2.0.0", bg=SURFACE, fg=MUTED,
                 font=("Courier",9)).pack(side="right", padx=20)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(body, bg=SURFACE, width=STYLE["sidebar_width"])
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        sb_canvas = tk.Canvas(sidebar, bg=SURFACE, highlightthickness=0)
        sb_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=sb_scroll.set)
        sb_scroll.pack(side="right", fill="y")
        sb_canvas.pack(fill="both", expand=True)
        sb_inner = tk.Frame(sb_canvas, bg=SURFACE)
        sb_canvas.create_window((0,0), window=sb_inner, anchor="nw")
        sb_inner.bind("<Configure>",
                      lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))

        section_label(sb_inner, "SEQUENCE").pack(anchor="w", padx=16, pady=(14,2))
        self.seq_panel = SequencePanel(sb_inner, on_transcript=self._on_transcript,
                                       on_dirty=self._on_dirty)
        self.seq_panel.pack(fill="x")

        tk.Frame(sb_inner, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        section_label(sb_inner, "SIMULATION PARAMETERS").pack(anchor="w", padx=16, pady=(0,4))
        self.param_panel = ParamPanel(sb_inner, on_change=self._render,
                                      on_dirty=self._on_dirty)
        self.param_panel.pack(fill="x", padx=8)

        tk.Frame(sb_inner, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        section_label(sb_inner, "ACTIONS").pack(anchor="w", padx=16, pady=(0,6))

        bf = tk.Frame(sb_inner, bg=SURFACE)
        bf.pack(fill="x", padx=10)
        styled_button(bf,"↓ SVG",self._download_svg,style="secondary").pack(side="left",expand=True,fill="x",padx=(0,4))
        styled_button(bf,"↓ PNG",self._download_png,style="secondary").pack(side="left",expand=True,fill="x")

        bf2 = tk.Frame(sb_inner, bg=SURFACE)
        bf2.pack(fill="x", padx=10, pady=(6,0))
        styled_button(bf2,"↑ Load Session",  self._load_json,   style="secondary").pack(fill="x",pady=(0,4))
        styled_button(bf2,"↓ Save Session",  self._save_json,   style="secondary").pack(fill="x",pady=(0,4))
        # styled_button(bf2,"↑ Import Events CSV",self._import_csv,style="secondary").pack(fill="x")

        # Main area
        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        # Events header
        eh = tk.Frame(main, bg=BG)
        eh.pack(fill="x", padx=20, pady=(16,4))
        section_label(eh, "EVENTS", bg=BG).pack(side="left")
        styled_button(eh,"+ Add",      self._add_event,       style="primary").pack(side="right",padx=(4,0))
        styled_button(eh,"✕ Clear All",self._clear_all_events,style="danger" ).pack(side="right")
        styled_button(eh,"Sort", self._sort_events, style="secondary").pack(side="right", padx=(4,0))

        # Legend
        legend = tk.Frame(main, bg=BG)
        legend.pack(fill="x", padx=20, pady=(0,2))
        for col, txt in ((AUTO_FG," auto-detected   "),(TEXT," manual")):
            tk.Label(legend,text="■",bg=BG,fg=col,font=MONO_S).pack(side="left")
            tk.Label(legend,text=txt, bg=BG,fg=MUTED,font=MONO_S).pack(side="left")

        # Transcript info bar
        info_bar = tk.Frame(main, bg=BG)
        info_bar.pack(fill="x", padx=20, pady=(0,4))
        self._transcript_info_var = tk.StringVar(value="No transcript loaded.")
        tk.Label(info_bar, textvariable=self._transcript_info_var,
                 bg=BG, fg=MUTED, font=MONO_S).pack(side="left")
        tk.Label(info_bar, text="  length:", bg=BG, fg=MUTED, font=MONO_S).pack(side="left")
        self._length_var   = tk.StringVar(value="")
        self._length_entry = styled_entry(info_bar, width=7, textvariable=self._length_var)
        self._length_entry.config(validate="key",
                                   validatecommand=_int_validator(self._length_entry),
                                   state="disabled")
        self._length_entry.pack(side="left", padx=(2,6))
        self._length_entry.bind("<Return>",   lambda _e: self._resize_transcript())
        self._length_entry.bind("<FocusOut>", lambda _e: self._resize_transcript())

        # Event table frame (scrollable)
        rows_outer = tk.Frame(main, bg=SURFACE)
        rows_outer.pack(fill="x", padx=20)
        self._rows_canvas = tk.Canvas(rows_outer, bg=SURFACE, highlightthickness=0, height=220)
        rows_scroll = ttk.Scrollbar(rows_outer, orient="vertical",
                                    command=self._rows_canvas.yview)
        self._rows_canvas.configure(yscrollcommand=rows_scroll.set)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>",   self._on_mousewheel)
        self.bind_all("<Button-5>",   self._on_mousewheel)
        rows_scroll.pack(side="right", fill="y")
        self._rows_canvas.pack(fill="both", expand=True)
        self._rows_inner = tk.Frame(self._rows_canvas, bg=SURFACE)
        self._rows_canvas.create_window((0,0), window=self._rows_inner, anchor="nw")
        self._rows_inner.bind("<Configure>",
                              lambda e: self._rows_canvas.configure(
                                  scrollregion=self._rows_canvas.bbox("all")))
        self._build_table_header()

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", padx=0, pady=8)

        # Preview
        preview_header = tk.Frame(main, bg=BG)
        preview_header.pack(fill="x", padx=20, pady=(0,8))
        section_label(preview_header, "PREVIEW", bg=BG).pack(side="left")
        self.svg_canvas = SVGCanvas(main, on_svg_ready=self._store_svg)
        self.svg_canvas.pack(fill="both", expand=True, padx=20, pady=(0,16))
        self.svg_canvas.clear()

    # ── Event table header ────────────────────────────────────────────────────
    def _build_table_header(self):
        for c, col in enumerate(self._ET_LABELS + [""]):
            tk.Label(self._rows_inner, text=col.upper(), bg=SURFACE, fg=TEXT,
                     font=MONO, anchor="w", padx=8).grid(row=0, column=c, sticky="w", pady=(0,6))
        tk.Frame(self._rows_inner, bg=BORDER, height=1).grid(
            row=1, column=0, columnspan=len(self._ET_LABELS)+1, sticky="ew", pady=(0,4))

    # ── Event table: load ─────────────────────────────────────────────────────
    def _load_transcript_into_table(self, transcript: Transcript):
        self._clear_table_widgets()
        for pos in transcript.events.keys():
            for etype, edata in transcript.events[pos].items():
                if etype == "end":
                    continue
                self._append_row(
                    pos=pos, etype=etype,
                    prob=edata.get("probability", 0.0),
                    manual=bool(edata.get("_manual")),
                )

    def _clear_table_widgets(self):
        for row_ws in self._row_widgets:
            for w in row_ws:
                w.destroy()
        self._row_widgets = []

    # ── Event table: add row ──────────────────────────────────────────────────
    def _add_event(self):
        if self._transcript is None:
            length = self._ask_transcript_length()
            if length is None: return
            stub = Transcript("N"*length, id="<manual>",
                              description="Manually defined transcript")
            self._transcript = stub
            self._load_transcript_into_table(stub)
            self._update_transcript_info()

        if sum(len(v) for v in self._transcript.events.values()) >= MAX_EVENTS:
            messagebox.showinfo("Limit reached", f"Maximum {MAX_EVENTS} events allowed.")
            return

        # Find a free (pos, "initiation") slot
        occupied = {pos for pos, evts in self._transcript.events.items()
                    if "initiation" in evts}
        pos = len(self._transcript) // 2
        if pos in occupied:
            for c in range(1, len(self._transcript)+1):
                if c not in occupied:
                    pos = c; break

        self._transcript.events[pos]["initiation"] = {
            "probability": 1.0, "_manual": True}
        self._append_row(pos=pos, etype="initiation", prob=1.0, manual=True)
        self._render()

    # ── Event table: clear all ────────────────────────────────────────────────
    def _clear_all_events(self):
        if self._transcript is None: return
        if not messagebox.askyesno("Clear all events",
                                   "Remove all events from the transcript?"):
            return
        # Preserve end events
        end_events = {
            pos: {et: ed for et, ed in evts.items() if et == "end"}
            for pos, evts in self._transcript.events.items()
            if any(et == "end" for et in evts)
        }
        self._transcript.events.clear()
        for pos, evts in end_events.items():
            self._transcript.events[pos].update(evts)
        self._clear_table_widgets()
        self._render()

    def _sort_events(self):
        if self._transcript is None:
            return

        # Flatten events into list
        items = []
        for pos, evts in self._transcript.events.items():
            for etype, edata in evts.items():
                items.append((pos, etype, edata))

        # Sort by position, then type
        items.sort(key=lambda x: (x[0], x[1]))

        # Rebuild ordered dict
        new_events = defaultdict(lambda: defaultdict(defaultdict[str, float|str|bool]))
        for pos, etype, edata in items:
            new_events[pos][etype] = edata

        self._transcript.events = new_events

        # Reload UI
        self._load_transcript_into_table(self._transcript)
        self._render()

    # ── Event table: append one row ───────────────────────────────────────────
    def _append_row(self, pos: int, etype: str, prob: float, manual: bool):
        i   = len(self._row_widgets) + 2      # +2 for header rows
        fg  = TEXT if manual else AUTO_FG
        ws  = []

        # Position
        pos_var = tk.IntVar(value=pos)
        pos_e   = styled_entry(self._rows_inner, width=6, textvariable=pos_var)
        pos_e.config(fg=fg, validate="key", validatecommand=_int_validator(pos_e))
        if not manual: pos_e.config(state="disabled")
        pos_e.grid(row=i, column=0, padx=4, pady=3, sticky="w")
        ws.append(pos_e)

        # Event type
        type_var = tk.StringVar(value=etype)
        om = tk.OptionMenu(self._rows_inner, type_var, *EVENT_TYPES)
        om.config(bg=BG, fg=fg, activebackground=BORDER, activeforeground=TEXT,
                  highlightthickness=0, relief="flat", font=MONO_S, width=12)
        om["menu"].config(bg=BG, fg=TEXT, activebackground=BORDER)
        if not manual: om.config(state="disabled")
        om.grid(row=i, column=1, padx=4, pady=3, sticky="w")
        ws.append(om)

        # Probability
        prob_max = _event_constraints(etype)
        prob_var = tk.DoubleVar(value=min(prob, prob_max))
        prob_e   = styled_entry(self._rows_inner, width=6, textvariable=prob_var)
        prob_e.config(fg=fg, validate="key", validatecommand=_float_validator(prob_e))
        prob_e.grid(row=i, column=2, padx=4, pady=3, sticky="w")
        ws.append(prob_e)

        # Type-change → clamp prob
        def _on_type_change(*_, tv=type_var, pv=prob_var):
            pm = _event_constraints(tv.get())
            try:
                if pv.get() > pm: pv.set(pm)
            except tk.TclError: pv.set(pm)

        type_var.trace_add("write", _on_type_change)

        # Committed key — mutable so both write and delete always target
        # the current dict key, not the one the row was created with.
        committed_key: list = [pos, etype]

        btn = styled_button(self._rows_inner, "✕",
                            lambda ck=committed_key, w=ws: self._delete_row(ck, w),
                            style="danger")
        btn._committed_key = committed_key
        btn.grid(row=i, column=4, padx=4)
        ws.append(btn)
        self._row_widgets.append(ws)

        # Callbacks
        def on_edit(*_, pv=pos_var, tv=type_var, prv=prob_var,
                    ck=committed_key, m=manual):
            self._write_row(pv, tv, prv, ck, m, render=False)

        def on_commit(*_, pv=pos_var, tv=type_var, prv=prob_var, 
                      ck=committed_key, m=manual):
            self._write_row(pv, tv, prv, ck, m, render=True)

        for var in (pos_var, type_var, prob_var):
            var.trace_add("write", on_edit)
        type_var.trace_add("write", on_commit)   # type change always renders
        for entry in (pos_e, prob_e):
            entry.bind("<FocusOut>", on_commit)
            entry.bind("<Return>",   on_commit)

    # ── Event table: write row back to transcript ─────────────────────────────
    def _other_committed_keys(self, exclude: list) -> list[tuple]:
        keys = []
        for ws in self._row_widgets:
            btn = ws[-1]
            ck  = getattr(btn, "_committed_key", None)
            if ck is not None and ck is not exclude:
                keys.append((ck[0], ck[1]))
        return keys

    def _write_row(self, pos_var, type_var, prob_var,
                   committed_key: list, manual: bool, render: bool = False):
        if self._transcript is None: return
        try:
            new_pos  = int(pos_var.get())
            new_type = type_var.get()
            new_prob = float(prob_var.get())
        except (tk.TclError, ValueError):
            return

        old_pos, old_type = committed_key

        # Only remove old key if no sibling row still owns it
        if (old_pos, old_type) not in self._other_committed_keys(committed_key):
            if old_pos in self._transcript.events:
                self._transcript.events[old_pos].pop(old_type, None)
                if not self._transcript.events[old_pos]:
                    del self._transcript.events[old_pos]

        self._transcript.events[new_pos][new_type] = {
            "probability": new_prob, "_manual": manual}

        committed_key[0] = new_pos
        committed_key[1] = new_type

        if render:
            self._render()
            self.after_idle(lambda: self._load_transcript_into_table(self._transcript))
        else:
            self._on_dirty()

    # ── Event table: delete row ───────────────────────────────────────────────
    def _delete_row(self, committed_key: list, widgets: list):
        pos, etype = committed_key
        if self._transcript and pos in self._transcript.events:
            self._transcript.events[pos].pop(etype, None)
            if not self._transcript.events[pos]:
                del self._transcript.events[pos]
        for w in widgets: w.destroy()
        if widgets in self._row_widgets: self._row_widgets.remove(widgets)
        self._render()

    # ── Transcript info bar ───────────────────────────────────────────────────
    def _update_transcript_info(self):
        t = self._transcript
        if t is None:
            self._transcript_info_var.set("No transcript loaded.")
            self._length_var.set("")
            self._length_entry.config(state="disabled")
        else:
            kind = "stub" if t.id == "<manual>" else t.id
            self._transcript_info_var.set(f"{kind}  |  pos 1 –")
            self._length_var.set(str(len(t)))
            self._length_entry.config(state="normal")

    def _resize_transcript(self):
        if self._transcript is None: return
        try:
            new_len = int(self._length_var.get())
            if new_len < 1: raise ValueError
        except (ValueError, tk.TclError):
            self._length_var.set(str(len(self._transcript))); return
        if new_len == len(self._transcript): return
        t     = self._transcript
        fresh = Transcript("N"*new_len, id=t.id, description=t.description)
        for pos, evts in t.events.items():
            if 1 <= pos <= new_len:
                for etype, edata in evts.items():
                    if etype not in ("cap","end"):
                        fresh.events[pos][etype] = edata
        self._transcript = fresh
        self._load_transcript_into_table(fresh)
        self._update_transcript_info()
        self._render()

    def _ask_transcript_length(self) -> int | None:
        win = tk.Toplevel(self)
        win.title("Transcript length"); win.configure(bg=SURFACE)
        win.resizable(False, False); win.grab_set()
        tk.Label(win, text="No sequence loaded.\nEnter a transcript length:",
                 bg=SURFACE, fg=TEXT, font=MONO_S, justify="left").pack(padx=20, pady=(16,8))
        var   = tk.IntVar(value=1000)
        entry = styled_entry(win, width=10, textvariable=var)
        entry.config(validate="key", validatecommand=_int_validator(entry))
        entry.pack(padx=20, pady=(0,12)); entry.focus_set(); entry.select_range(0,"end")
        result: list[int|None] = [None]
        def confirm(_e=None):
            try:
                v = int(entry.get())
                if v < 1: raise ValueError
                result[0] = v; win.destroy()
            except (ValueError, tk.TclError):
                entry.config(highlightbackground=DANGER)
        def cancel(): win.destroy()
        br = tk.Frame(win, bg=SURFACE); br.pack(padx=20, pady=(0,16))
        styled_button(br,"OK",    confirm,style="primary"  ).pack(side="left",padx=(0,8))
        styled_button(br,"Cancel",cancel, style="secondary").pack(side="left")
        entry.bind("<Return>", confirm); win.bind("<Escape>", lambda _e: cancel())
        win.wait_window(); return result[0]

    # ── App callbacks ─────────────────────────────────────────────────────────
    def _on_transcript(self, transcript: Transcript):
        self._transcript = transcript
        self._load_transcript_into_table(transcript)
        self._update_transcript_info()
        self._render()

    def _on_dirty(self):
        self.svg_canvas.clear("Modified — press Enter or click away to render")

    def _store_svg(self, svg_text): self._last_svg = svg_text

    def _on_mousewheel(self, event):
        delta = -1 if event.num==4 else (1 if event.num==5 else int(-event.delta/120))
        w = event.widget
        while w is not None:
            if w is self._rows_canvas:
                self._rows_canvas.yview_scroll(delta,"units"); return
            w = getattr(w,"master",None)

    # ── Render ────────────────────────────────────────────────────────────────
    def _render(self):
        if self._transcript is None:
            self.svg_canvas.clear("Load a sequence or add events to render."); return
        transcript = self._transcript
        params     = self.param_panel.get_values()
        self.svg_canvas.show_loading()
        self._render_gen += 1
        my_gen = self._render_gen
        def do_render():
            try:
                svg_text, png_bytes = render_transcript(transcript, params)
                if my_gen != self._render_gen: return
                if isinstance(svg_text, str) and ("<svg" in svg_text or "<?xml" in svg_text):
                    self.after(0, lambda s=svg_text,p=png_bytes: self.svg_canvas.load_svg(s,p))
                else:
                    msg = svg_text if isinstance(svg_text,str) else "Unknown error"
                    self.after(0, lambda m=msg: self.svg_canvas.clear(f"⚠ {m}"))
            except Exception as e:
                traceback.print_exc()
                if my_gen == self._render_gen:
                    self.after(0, lambda m=str(e): self.svg_canvas.clear(f"Error: {m}"))
        threading.Thread(target=do_render, daemon=True).start()

    # ── Session save / load ───────────────────────────────────────────────────
    def _save_json(self):
        if self._transcript is None:
            messagebox.showwarning("Nothing to save","Load a transcript first."); return
        path = filedialog.asksaveasfilename(defaultextension=".json",
            filetypes=[("JSON","*.json"),("All","*.*")], initialfile="session.json")
        if not path: return
        t = self._transcript
        data = {
            "transcript": {"id":t.id,"description":t.description,"sequence":str(t.seq)},
            "events": {
                str(pos): {etype: {k:v for k,v in edata.items()}
                           for etype, edata in evts.items()}
                for pos, evts in t.events.items()},
            "params": self.param_panel.get_values(),
        }
        try:
            with open(path,"w") as f: json.dump(data,f,indent=2)
        except OSError as e:
            messagebox.showerror("Save failed",str(e))

    def _load_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        try:
            with open(path) as f: data = json.load(f)
        except (OSError,json.JSONDecodeError) as e:
            messagebox.showerror("Load failed",str(e)); return
        try:
            td = data["transcript"]
            t  = Transcript(td["sequence"],id=td.get("id","<unknown>"),
                            description=td.get("description",""))
            for pos_str, evts in data.get("events",{}).items():
                for etype, edata in evts.items():
                    t.events[int(pos_str)][etype] = edata
            self.param_panel.set_values(data.get("params",{}))
            self._on_transcript(t)
        except Exception as e:
            messagebox.showerror("Load failed",f"Invalid session file:\n{e}")


    def _download_svg(self):
        if not self._last_svg:
            messagebox.showinfo("No data","Render the graph first."); return
        path = filedialog.asksaveasfilename(defaultextension=".svg",
            filetypes=[("SVG","*.svg"),("All","*.*")], initialfile="output.svg")
        if path:
            with open(path,"w") as f: f.write(self._last_svg)

    def _download_png(self):
        png = self.svg_canvas._png_bytes
        if not png:
            messagebox.showinfo("No data","Render the graph first."); return
        path = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("All","*.*")], initialfile="output.png")
        if path:
            with open(path,"wb") as f: f.write(png)

    def _show_errors(self, errors):
        win = tk.Toplevel(self); win.title("Errors"); win.configure(bg=SURFACE)
        win.geometry("540x320")
        tk.Label(win,text=f"{len(errors)} error(s):",bg=SURFACE,fg=DANGER,font=MONO).pack(
            anchor="w",padx=16,pady=(12,4))
        frame=tk.Frame(win,bg=SURFACE); frame.pack(fill="both",expand=True,padx=16,pady=(0,8))
        sb=ttk.Scrollbar(frame); txt=tk.Text(frame,bg=BG,fg=TEXT,font=MONO_S,relief="flat",
            highlightthickness=0,wrap="word",yscrollcommand=sb.set)
        sb.config(command=txt.yview); sb.pack(side="right",fill="y"); txt.pack(fill="both",expand=True)
        for e in errors: txt.insert("end",f"• {e}\n")
        txt.config(state="disabled")
        styled_button(win,"Close",win.destroy).pack(pady=(0,12))

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()
