"""
Dyno RDG — Tkinter desktop UI
Replaces the web frontend entirely. No ports, no browser needed.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import io
import math

# Pre-import heavy packages in main thread to avoid numpy threading issues
import numpy
import pandas
import networkx
import matplotlib

# ── Colours (matching the web UI palette) ──────────────────────────────────
BG       = "#0e0f11"
SURFACE  = "#16181c"
BORDER   = "#2a2d35"
ACCENT   = "#00e5a0"
ACCENT2  = "#0077ff"
DANGER   = "#ff4455"
TEXT     = "#e2e4ea"
MUTED    = "#5a5f72"

STYLE = {
    "font_mono":        ("Courier", 11),
    "font_mono_small":  ("Courier", 9),
    "font_mono_tiny":   ("Courier", 8),
    "font_header":      ("Courier", 12, "bold"),
    "font_version":     ("Courier", 9),
    "font_sans":        ("Hevetica", 11),
    "sidebar_width":    260,
    "rows_height":      200,
    "window_width":     1280,
    "window_height":    820,
}

# ── Font helpers ────────────────────────────────────────────────────────────
MONO    = STYLE['font_mono']
MONO_S  = STYLE['font_mono_small']
SANS    = STYLE['font_sans']


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def blend(fg, bg, alpha):
    fr, fg_, fb = hex_to_rgb(fg)
    br, bg_, bb = hex_to_rgb(bg)
    return rgb_to_hex(
        int(br + (fr - br) * alpha),
        int(bg_ + (fg_ - bg_) * alpha),
        int(bb + (fb - bb) * alpha),
    )

def get_label(section, key):
    """Get display label for a key, falling back to the key itself."""
    return SCHEMA.get("labels", {}).get(section, {}).get(key, key.replace("_", " ").title())


# ── Schema ──────────────────────────────────────────────────────────────────
# Mirrors what /schema returns from FastAPI. Edit to match your actual schema.
SCHEMA = {
    "params": {
        "transcript_length": {"type": "int",   "default": 1000, "min": 1},
        "loading":           {"type": "float", "default": 1, "min": 0, "max": 10, "step": 0.01},
        "log_reduction":     {"type": "float", "default": 1.5, "min": 1, "max": 10, "step": 0.1},
        "height_scale":      {"type": "float", "default": 2.0, "min": 0.5, "max": 10, "step": 0.1},
        "bulk_length":       {"type": "float", "default": 0.3, "min": 0, "max": 5,  "step": 0.1},
    },
    "row": {
        "pos":       {"type": "int",    "default": 100,      "min": 1, "max_param": "transcript_length"},
        "type":      {"type": "select", "default": "init",  "options": ["stop", "init", "ires", "shift+1", "shift-1"]},
        "prob":      {"type": "float",  "default": 1,    "min": 0, "max": 1, "step": 0.01},
        "drop_prob": {"type": "float",  "default": 0.0,    "min": 0, "max": 1, "step": 0.01},
    },
    # ── Display labels — edit these freely without touching anything else ──
    "labels": {
        "params": {
            "transcript_length": "Transcript Length",
            "loading":           "Loading Rate",
            "log_reduction":     "Log Reduction",
            "height_scale":      "Height Scale",
            "bulk_length":       "Bulk Length",
        },
        "row": {
            "pos":       "Position",
            "type":      "Event Type",
            "prob":      "Probability",
            "drop_prob": "Drop Probability",
        }
    }
}


# ── Styled widgets ───────────────────────────────────────────────────────────

def styled_entry(parent, width=8, **kw):
    e = tk.Entry(
        parent,
        width=width,
        bg=BG,
        fg=TEXT,
        insertbackground=ACCENT,
        relief="flat",
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
        font=MONO_S,
        **kw
    )
    return e


def styled_button(parent, text, command, style="primary", **kw):
    colours = {
        "primary":   (ACCENT,  "#000000"),
        "secondary": (SURFACE, ACCENT2),
        "danger":    (SURFACE, DANGER),
    }
    bg, fg = colours.get(style, (ACCENT, "#000"))

    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=blend(bg, "#ffffff", 0.15),
        activeforeground=fg,
        relief="flat",
        font=MONO_S,
        cursor="hand2",
        padx=10,
        pady=4,
        **kw
    )
    if style == "secondary":
        btn.config(highlightthickness=1, highlightbackground=ACCENT2)
    if style == "danger":
        btn.config(highlightthickness=1, highlightbackground=DANGER)
    return btn


def label(parent, text, muted=False, **kw):
    return tk.Label(
        parent,
        text=text,
        bg=kw.pop("bg", SURFACE),
        fg=MUTED if muted else TEXT,
        font=MONO_S,
        **kw
    )


# ── SVG Canvas ───────────────────────────────────────────────────────────────

class SVGCanvas(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=SURFACE, highlightthickness=0, **kw)
        self._png_bytes = None
        self._photo = None
        self._svg_text = None  # keep for download
        self.bind("<Configure>", self._on_resize)

    def load_svg(self, svg_text, png_bytes=None):
        self._svg_text = svg_text
        self._png_bytes = png_bytes
        # Store on parent app for download
        app = self.winfo_toplevel()
        if hasattr(app, '_last_svg'):
            app._last_svg = svg_text
        self._draw()

    def _on_resize(self, event):
        if self._png_bytes:
            self._draw()

    def _draw(self):
        if not self._png_bytes:
            print("DEBUG: _png_bytes is None or empty")
            return
        try:
            import io
            from PIL import Image, ImageTk

            print(f"DEBUG: png_bytes length = {len(self._png_bytes)}")

            cw = max(self.winfo_width(), 100)
            ch = max(self.winfo_height(), 100)
            print(f"DEBUG: canvas size = {cw}x{ch}")

            img = Image.open(io.BytesIO(self._png_bytes))
            print(f"DEBUG: image size = {img.size}")

            img.thumbnail((cw, ch), Image.LANCZOS)
            print(f"DEBUG: thumbnail size = {img.size}")

            self.delete("all")
            photo = ImageTk.PhotoImage(img)
            self._photo = photo
            x = (cw - img.width) // 2
            y = (ch - img.height) // 2
            print(f"DEBUG: placing image at {x},{y}")
            self.create_image(x, y, anchor="nw", image=photo)
            print("DEBUG: image created successfully")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.delete("all")
            self.create_text(
                self.winfo_width() // 2 or 400,
                self.winfo_height() // 2 or 200,
                text=f"Display error:\n{e}",
                fill=DANGER,
                font=MONO_S,
            )

    def clear(self, message="No events yet — add  events to render"):
        self._png_bytes = None
        self._photo = None
        self.delete("all")
        self.create_text(
            max(self.winfo_width() // 2, 400),
            max(self.winfo_height() // 2, 200),
            text=message,
            fill=DANGER,
            font=MONO,
        )

# ── Row table ────────────────────────────────────────────────────────────────

class RowTable(tk.Frame):
    def __init__(self, parent, schema, on_change, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.schema    = schema["row"]
        self.on_change = on_change
        self.rows      = []
        self._widgets  = []
        self._build_header()

    def _build_header(self):
        cols = [get_label("row", k) for k in self.schema.keys()] + [""]
        for c, col in enumerate(cols):
            tk.Label(
                self,
                text=col.upper(),
                bg=SURFACE,
                fg=TEXT,
                font=(STYLE['font_mono']),
                anchor="w",
                padx=8,
            ).grid(row=0, column=c, sticky="w", pady=(0, 6))
        # separator
        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.grid(row=1, column=0, columnspan=len(cols), sticky="ew", pady=(0, 4))

    def add_row(self):
        row = {k: v["default"] for k, v in self.schema.items()}
        self.rows.append(row)
        self._redraw()
        self.on_change()

    def delete_row(self, i):
        self.rows.pop(i)
        self._redraw()
        self.on_change()

    def _redraw(self):
        # Clear old widgets
        for w in self._widgets:
            w.destroy()
        self._widgets = []

        for i, row in enumerate(self.rows):
            col = 0
            for key, spec in self.schema.items():
                w = self._make_cell(i, key, spec, row[key])
                w.grid(row=i + 2, column=col, padx=4, pady=3, sticky="w")
                self._widgets.append(w)
                col += 1

            # Delete button
            btn = styled_button(self, "✕", lambda i=i: self.delete_row(i), style="danger")
            btn.grid(row=i + 2, column=col, padx=4)
            self._widgets.append(btn)

    def _make_cell(self, row_i, key, spec, value):
        t = spec["type"]

        if t == "select":
            var = tk.StringVar(value=value)
            def on_select(v, ri=row_i, k=key, sv=var):
                self.rows[ri][k] = sv.get()
                self.on_change()
            om = tk.OptionMenu(self, var, *spec["options"], command=on_select)
            om.config(
                bg=BG, fg=TEXT, activebackground=BORDER,
                activeforeground=TEXT, highlightthickness=0,
                relief="flat", font=MONO_S, width=8
            )
            om["menu"].config(bg=BG, fg=TEXT, activebackground=BORDER)
            return om

        if t == "float":
            frame = tk.Frame(self, bg=SURFACE)
            var = tk.DoubleVar(value=value)

            entry = styled_entry(frame, width=6, textvariable=var)
            entry.pack(side="left", padx=(0, 4))

            slider = tk.Scale(
                frame,
                from_=spec.get("min", 0),
                to=spec.get("max", 1),
                resolution=spec.get("step", 0.01),
                orient="horizontal",
                variable=var,
                bg=SURFACE,
                fg=TEXT,
                troughcolor=BORDER,
                activebackground=ACCENT,
                highlightthickness=0,
                sliderrelief="flat",
                length=100,
                showvalue=False,
                font=MONO_S,
            )
            slider.pack(side="left")

            def on_float_change(*_, ri=row_i, k=key, v=var):
                val = v.get()
                # Clamp prob + drop_prob <= 1
                if k in ("prob", "drop_prob"):
                    other = "drop_prob" if k == "prob" else "prob"
                    other_val = self.rows[ri].get(other, 0)
                    if val + other_val > 1:
                        val = round(1 - other_val, 10)
                        v.set(val)
                self.rows[ri][k] = val
                self.on_change()

            var.trace_add("write", on_float_change)
            return frame

        if t == "int":
            var = tk.IntVar(value=value)
            entry = styled_entry(self, width=6, textvariable=var)

            def on_int_change(*_, ri=row_i, k=key, v=var, s=spec):
                try:
                    val = v.get()
                    mn = s.get("min", 0)
                    mx = s.get("max", 9999)
                    val = max(mn, min(mx, val))
                    self.rows[ri][k] = val
                    self.on_change()
                except tk.TclError:
                    pass

            var.trace_add("write", on_int_change)
            return entry

        # fallback text
        var = tk.StringVar(value=str(value))
        entry = styled_entry(self, textvariable=var)
        var.trace_add("write", lambda *_, ri=row_i, k=key, v=var: (
            self.rows[ri].update({k: v.get()}), self.on_change()
        ))
        return entry


# ── Param panel ──────────────────────────────────────────────────────────────

class ParamPanel(tk.Frame):
    def __init__(self, parent, schema, on_change, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self.schema    = schema["params"]
        self.on_change = on_change
        self._vars     = {}
        self._build()

    def _build(self):
        for k, spec in self.schema.items():
            tk.Label(
                self, text=get_label("params", k) + ":",
                bg=SURFACE, fg=TEXT,
                font=STYLE['font_mono'], anchor="w"
            ).pack(fill="x", padx=4, pady=(6, 0))
            # ... rest of method unchanged

            t = spec["type"]
            if t in ("int", "float"):
                var = tk.DoubleVar(value=spec["default"])
                entry = styled_entry(self, textvariable=var)
                entry.pack(fill="x", padx=4, pady=(2, 0))
                var.trace_add("write", lambda *_, v=var, k=k: self._on_param(k, v))
                self._vars[k] = var
            else:
                var = tk.StringVar(value=spec["default"])
                entry = styled_entry(self, textvariable=var)
                entry.pack(fill="x", padx=4, pady=(2, 0))
                var.trace_add("write", lambda *_, v=var, k=k: self._on_param(k, v))
                self._vars[k] = var

    def _on_param(self, key, var):
        self.on_change()

    def get_values(self):
        out = {}
        for k, spec in self.schema.items():
            try:
                v = self._vars[k].get()
                out[k] = int(v) if spec["type"] == "int" else float(v) if spec["type"] == "float" else v
            except tk.TclError:
                out[k] = spec["default"]
        return out


# ── Main application ─────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dyno RDG")
        self.geometry("1280x820")
        self.configure(bg=BG)
        self.minsize(900, 600)

        self._render_timer = None
        self._last_svg     = None

        self._build_ui()
    def _on_rows_mousewheel(self, event):
        self._rows_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=SURFACE, height=48)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(
            header, text="DYNO  RDG",
            bg=SURFACE, fg=ACCENT,
            font=STYLE['font_header'],
        ).pack(side="left", padx=20)

        tk.Label(
            header, text="v1.0.7",
            bg=SURFACE, fg=MUTED,
            font=STYLE['font_version'],
        ).pack(side="right", padx=20)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x")

        # ── Body ─────────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # ── Left sidebar ─────────────────────────────────────────────────────
        sidebar = tk.Frame(body, bg=SURFACE, width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        label(sidebar, "PARAMETERS", muted=True, bg=SURFACE).pack(
            anchor="w", padx=16, pady=(16, 4)
        )

        self.param_panel = ParamPanel(
            sidebar, SCHEMA, on_change=self._schedule_render
        )
        self.param_panel.pack(fill="x", padx=8)

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=16, pady=12)

        label(sidebar, "ACTIONS", muted=True, bg=SURFACE).pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        btn_frame = tk.Frame(sidebar, bg=SURFACE)
        btn_frame.pack(fill="x", padx=12)

        styled_button(btn_frame, "+ Add Event", self._add_row).pack(side="left", padx=(0, 6))
        styled_button(btn_frame, "↓ SVG", self._download_svg, style="secondary").pack(side="left", padx=(0, 6))
        styled_button(btn_frame, "↓ PNG", self._download_png, style="secondary").pack(side="left")

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", padx=16, pady=12)

        label(sidebar, "INSTRUCTIONS - Scroll to see all", muted=False, bg=SURFACE).pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        instructions_text = """Add rows to define events along your transcript.

Position: nucleotide position of the event.

Event Type:
stop  — termination codon
init  — start codon
ires  — internal ribosome entry
shift — frameshift (+1, -1, etc.)

Probability: chance the event occurs.

Drop Probability: chance ribosome dissociates from transcript.

For STOP events, Probality is the chance of scanning occuring i.e. leading to reinitiation, while \
drop probability is the likelihood of the STOP acting as a normal stop codon

Loading Rate controls how many ribosomes are loaded onto the transcript (set in Parameters).

Adjust Log Reduction to compress the distance between events logarithmically, \
i.e. longer distances get compressed by a greater factor than smaller distances.

Bulk Length controls the relative length of the bulk (purple) edges compared to a ribosomal flux of 1

Height Scale controls the vertical scaling of the graph, namely the distance between states, not height relative to length"""

        instructions = tk.Text(
            sidebar,
            bg=SURFACE,
            fg=TEXT,
            font=("Courier", 10),
            relief="flat",
            wrap="word",
            height=18,
            padx=8,
            pady=4,
            highlightthickness=0,
            state="normal",
            cursor="arrow",
        )
        instructions.insert("1.0", instructions_text)
        instructions.config(state="disabled")  # read only
        instructions.pack(fill="x", padx=8, pady=(0, 16))

        # ── Main area ─────────────────────────────────────────────────────────
        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        # Rows section
        rows_header = tk.Frame(main, bg=BG)
        rows_header.pack(fill="x", padx=20, pady=(16, 4))
        label(rows_header, "EVENTS", muted=True, bg=BG).pack(side="left")

        

        rows_outer = tk.Frame(main, bg=SURFACE, bd=0)
        rows_outer.pack(fill="x", padx=20, pady=(0, 0))

        # Both scrollbars
        self._rows_canvas = tk.Canvas(
            rows_outer, bg=SURFACE, highlightthickness=0, height=200
        )

        rows_scroll_y = ttk.Scrollbar(
            rows_outer, orient="vertical", command=self._rows_canvas.yview
        )
        self._rows_canvas.configure(
            yscrollcommand=rows_scroll_y.set
        )

        self._rows_canvas.bind("<Enter>", lambda e: self._rows_canvas.bind_all("<MouseWheel>", self._on_rows_mousewheel))
        self._rows_canvas.bind("<Leave>", lambda e: self._rows_canvas.unbind_all("<MouseWheel>"))
    


    
        rows_scroll_y.pack(side="right", fill="y")
        self._rows_canvas.pack(fill="both", expand=True)

        self._rows_inner = tk.Frame(self._rows_canvas, bg=SURFACE)
        self._rows_canvas_window = self._rows_canvas.create_window(
            (0, 0), window=self._rows_inner, anchor="nw"
        )
        self._rows_inner.bind(
            "<Configure>",
            lambda e: self._rows_canvas.configure(
                scrollregion=self._rows_canvas.bbox("all")
            )
        )

        self.row_table = RowTable(
            self._rows_inner, SCHEMA, on_change=self._schedule_render
        )
        self.row_table.pack(padx=12, pady=8, anchor="nw")

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x", padx=0, pady=8)

        # Preview section
        preview_header = tk.Frame(main, bg=BG)
        preview_header.pack(fill="x", padx=20, pady=(0, 8))
        label(preview_header, "PREVIEW", muted=True, bg=BG).pack(side="left")

        self.svg_canvas = SVGCanvas(main)
        self.svg_canvas.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.svg_canvas.clear()

        self.after(100, self._add_row)

    def _add_row(self):
        self.row_table.add_row()

    def _schedule_render(self, delay=150):
        if self._render_timer:
            self.after_cancel(self._render_timer)
        self._render_timer = self.after(delay, self._render)

    def _render(self):
        rows   = self.row_table.rows
        params = self.param_panel.get_values()
        
        def do_render():
            try:
                from backend.render import render_svg
                import copy

                # Always render — even with no rows, loading alone may produce output
                result = render_svg(copy.deepcopy(rows), params)

                if isinstance(result, tuple):
                    svg_text, png_bytes = result
                else:
                    svg_text, png_bytes = result, None

                if isinstance(svg_text, str) and ('<?xml' in svg_text or '<svg' in svg_text):
                    self.after(0, lambda s=svg_text, p=png_bytes: self.svg_canvas.load_svg(s, p))
                else:
                    msg = svg_text if isinstance(svg_text, str) else "Unknown error"
                    self.after(0, lambda m=msg: self.svg_canvas.clear(f"⚠ {m}"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda m=str(e): self.svg_canvas.clear(f"Error: {m}"))

        
        threading.Thread(target=do_render, daemon=True).start()

    def _download_svg(self):
        if not self._last_svg:
            # trigger a render first
            rows   = self.row_table.rows
            params = self.param_panel.get_values()
            if not rows:
                messagebox.showinfo("No data", "Add events before downloading.")
                return
            try:
                from backend.render import render_svg
                result = render_svg(rows, params)
                if isinstance(result, str) and result.startswith("<svg"):
                    self._last_svg = result
                else:
                    messagebox.showerror("Error", str(result))
                    return
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return

        path = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
            initialfile="output.svg",
        )
        if path:
            with open(path, "w") as f:
                f.write(self._last_svg)

    def _download_png(self):
        png_bytes = self.svg_canvas._png_bytes
        if not png_bytes:
            messagebox.showinfo("No data", "Add events and render before downloading.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile="output.png",
        )
        if path:
            with open(path, "wb") as f:
                f.write(png_bytes)



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Force numpy/pandas to initialize in main thread before any background threads
    import numpy
    import pandas
    import networkx
    import matplotlib
    matplotlib.use('Agg')

    app = App()
    app.mainloop()
