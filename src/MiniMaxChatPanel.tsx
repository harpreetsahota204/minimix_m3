import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useOperatorExecutor } from "@fiftyone/operators";
import { useModalSample, useRefreshSample } from "@fiftyone/state";
import { usePanelClient } from "./hooks/usePanelClient";
import type { PanelData, PanelSchema, SaveLabelResult, Turn } from "./types";

// Operator that writes a converted label and refreshes the open modal.
const SAVE_LABEL_URI = "@harpreetsahota/minimax-m3/save_minimax_label";

// Image encode resolution (longest side, px) sent to M3. Higher = more detail
// for small objects at the cost of a larger payload; 0 = native (no downscale).
// Mirrors minimax_api.DEFAULT_IMAGE_MAX_SIDE.
const DEFAULT_IMAGE_MAX_SIDE = 1280;
const IMAGE_RESOLUTIONS: { label: string; value: number }[] = [
  { label: "Standard", value: 1280 },
  { label: "High",     value: 2048 },
  { label: "Native",   value: 0 },
];

// M3 reasoning modes, mirroring the operator form's thinking control (minus the
// per-task "auto", which has no meaning in free-form chat). Off is the default
// for the cleanest JSON.
const DEFAULT_THINKING = "disabled";
const THINKING_MODES: { label: string; value: string }[] = [
  { label: "Off",      value: "disabled" },
  { label: "Adaptive", value: "adaptive" },
  { label: "Always",   value: "enabled" },
];

// Generation-parameter defaults. Mirror the Python panel defaults
// (chat_panel._PANEL_DEFAULT_MAX_TOKENS and minimax_api M3-recommended sampling).
const DEFAULT_MAX_TOKENS  = 1500;
const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_TOP_P       = 0.95;
const DEFAULT_TOP_K       = 40;

// ---------------------------------------------------------------------------
// Logging — prefixed for easy filtering in DevTools ([minimax_chat])
// ---------------------------------------------------------------------------

function log(...args: unknown[]) {
  console.log("[minimax_chat]", ...args);
}
function logError(...args: unknown[]) {
  console.error("[minimax_chat]", ...args);
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

// ---------------------------------------------------------------------------
// Session persistence — per-sample in sessionStorage
// ---------------------------------------------------------------------------

const SESSION_PREFIX = "minimaxChat:";

interface StoredSession {
  turns: Turn[];
  thinking: string;
  hintFormat: string;
  nFrames: number;
  /** Image encode resolution (longest side, px); 0 = native. */
  imageMaxSide?: number;
  /** Per-format edits to the format instruction; empty unless customized. */
  hintTexts?: Record<string, string>;
  /** Sampling controls; absent = panel defaults. */
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

function loadSession(sampleId: string): StoredSession | null {
  try {
    const raw = sessionStorage.getItem(SESSION_PREFIX + sampleId);
    return raw ? (JSON.parse(raw) as StoredSession) : null;
  } catch {
    return null;
  }
}

function saveSession(sampleId: string, data: StoredSession): void {
  try {
    sessionStorage.setItem(SESSION_PREFIX + sampleId, JSON.stringify(data));
  } catch {}
}

// ---------------------------------------------------------------------------
// Per-turn save state — tracks the Convert to FiftyOne workflow per turn.
// ---------------------------------------------------------------------------

interface TurnSaveState {
  fieldName: string;
  saving: boolean;
  saved: boolean;
  savedLabelType?: string;
  savedCount?: number;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Global style injection (once per page load)
// ---------------------------------------------------------------------------

let _stylesInjected = false;
function ensureStyles() {
  if (_stylesInjected) return;
  _stylesInjected = true;
  const el = document.createElement("style");
  el.textContent = `
    @keyframes mmxSpin  { to { transform: rotate(360deg); } }
    @keyframes mmxBlink { 0%,100%{opacity:1} 50%{opacity:0} }
    .mmx-md { font-size: 13px; line-height: 1.7; color: var(--fo-palette-text-primary); word-break: break-word; }
    .mmx-md p  { margin: 0 0 8px; }
    .mmx-md p:last-child { margin-bottom: 0; }
    .mmx-md h1,.mmx-md h2,.mmx-md h3 { margin: 12px 0 5px; font-weight: 600; }
    .mmx-md ul,.mmx-md ol { margin: 0 0 8px; padding-left: 18px; }
    .mmx-md li { margin-bottom: 2px; }
    .mmx-md code { font-family: ui-monospace, monospace; font-size: 12px; background: var(--fo-palette-background-level2); color: var(--fo-palette-primary-main); padding: 1px 4px; border-radius: 3px; }
    .mmx-md pre { background: var(--fo-palette-background-level2); border: 1px solid var(--fo-palette-divider); border-radius: 4px; padding: 8px 10px; overflow-x: auto; margin: 0 0 8px; }
    .mmx-md pre code { background: none; padding: 0; color: var(--fo-palette-text-primary); }
    .mmx-md blockquote { margin: 0 0 8px; padding: 3px 10px; border-left: 3px solid var(--fo-palette-primary-main); color: var(--fo-palette-text-secondary); }
    .mmx-md strong { font-weight: 600; }
    .mmx-md em    { font-style: italic; }
    .mmx-md a     { color: var(--fo-palette-primary-main); }
  `;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------

const V = {
  bg:       "var(--fo-palette-background-body)",
  bg2:      "var(--fo-palette-background-level2)",
  bg3:      "var(--fo-palette-background-level3)",
  divider:  "var(--fo-palette-divider)",
  text:     "var(--fo-palette-text-primary)",
  muted:    "var(--fo-palette-text-secondary)",
  dim:      "var(--fo-palette-text-tertiary)",
  primary:  "var(--fo-palette-primary-main)",
  textInv:  "var(--fo-palette-text-invert)",
  font:     "var(--fo-fontFamily-body)",
  red:      "#e08080",
  redBg:    "#2a0e0e",
  green:    "#4caf81",
  greenBg:  "#0b1a0b",
  greenBorder: "#1a3a1a",
};

// Human-readable label for each grounding format.
const FORMAT_LABELS: Record<string, string> = {
  box:      "fo.Detections (boxes)",
  point:    "fo.Keypoints",
  temporal: "fo.TemporalDetections",
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type StreamState = "idle" | "running" | "done" | "error";

interface Props {
  data?:   PanelData;
  schema?: PanelSchema;
}

// ---------------------------------------------------------------------------
// GenParam — a compact labelled number input for a generation parameter.
// ---------------------------------------------------------------------------

interface GenParamProps {
  label:    string;
  title:    string;
  value:    number;
  width:    number;
  min?:     number;
  max?:     number;
  step?:    number;
  onChange: (n: number) => void;
}

const GenParam: React.FC<GenParamProps> = ({
  label, title, value, width, min, max, step, onChange,
}) => (
  <label title={title} style={{ display: "flex", alignItems: "center",
                                gap: 6, fontSize: 11, color: V.muted }}>
    <span style={{ flexShrink: 0, minWidth: 72, textAlign: "right" as const }}>{label}:</span>
    <input
      type="number" value={value} min={min} max={max} step={step}
      onChange={(e) => {
        const n = Number(e.target.value);
        if (!Number.isNaN(n)) onChange(n);
      }}
      style={{
        width, background: V.bg2, color: V.text,
        border: `1px solid ${V.divider}`, borderRadius: 4,
        padding: "2px 5px", fontSize: 11, fontFamily: V.font, outline: "none",
      }}
    />
  </label>
);

// ---------------------------------------------------------------------------
// RadioGroup — a compact labelled row of radio options (string or number).
// Shares GenParam's 72px right-aligned label column so they line up.
// ---------------------------------------------------------------------------

interface RadioOption<T> {
  label: string;
  value: T;
}

interface RadioGroupProps<T> {
  label:    string;
  title:    string;
  name:     string;
  value:    T;
  options:  RadioOption<T>[];
  onChange: (value: T) => void;
}

function RadioGroup<T extends string | number>({
  label, title, name, value, options, onChange,
}: RadioGroupProps<T>) {
  return (
    <div title={title} style={{ display: "flex", alignItems: "center",
                                 gap: 6, fontSize: 11, color: V.muted }}>
      <span style={{ flexShrink: 0, minWidth: 72, textAlign: "right" as const }}>{label}:</span>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" as const }}>
        {options.map((o) => (
          <label key={String(o.value)}
                 style={{ display: "flex", alignItems: "center", gap: 4,
                          cursor: "pointer", userSelect: "none" }}>
            <input
              type="radio" name={name}
              checked={value === o.value}
              onChange={() => onChange(o.value)}
              style={{ width: 12, height: 12, margin: 0, cursor: "pointer",
                       accentColor: V.primary, flexShrink: 0 }}
            />
            {o.label}
          </label>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const MiniMaxChatPanel: React.FC<Props> = ({ data, schema }) => {
  const uris = {
    ask:              schema?.view?.ask              ?? "",
    get_stream_chunk: schema?.view?.get_stream_chunk ?? "",
  };
  const { ask, getStreamChunk } = usePanelClient(uris);
  const saveLabelExecutor = useOperatorExecutor(SAVE_LABEL_URI);
  const modalSample = useModalSample();
  const refreshSample = useRefreshSample();

  // Current sample context pushed by Python.
  const [filepath,  setFilepath]  = useState("");
  const [sampleId,  setSampleId]  = useState("");
  const [mediaType, setMediaType] = useState("image");
  const [frameRate, setFrameRate] = useState<number | null>(null);

  // Conversation turns.
  const [turns,    setTurns]    = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  // M3 reasoning mode: "disabled" | "adaptive" | "enabled".
  const [thinking, setThinking] = useState(DEFAULT_THINKING);
  // "auto" = let the model decide; other values append a JSON-shape suffix.
  const [hintFormat, setHintFormat] = useState("auto");
  const [nFrames, setNFrames] = useState(8);
  // Image encode resolution (longest side, px); 0 = native (no downscale).
  const [imageMaxSide, setImageMaxSide] = useState(DEFAULT_IMAGE_MAX_SIDE);
  // Per-format edits to the format instruction; absence = use the default.
  const [hintTexts, setHintTexts] = useState<Record<string, string>>({});
  const [hintEditing, setHintEditing] = useState(false);
  // Generation parameters (shown under the Advanced toggle).
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [topP, setTopP] = useState(DEFAULT_TOP_P);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Composer focus drives the input pill's focus ring.
  const [inputFocused, setInputFocused] = useState(false);

  // Format-instruction templates are owned by Python (prompts.JSON_SHAPE_BY_FORMAT)
  // and pushed via data.hint_templates — the single source of truth.
  const hintTemplates = useMemo(
    () => data?.hint_templates ?? {},
    [data?.hint_templates]
  );
  // The instruction text for the current output style (edited or default).
  const currentHintText =
    hintFormat === "auto" ? "" : (hintTexts[hintFormat] ?? hintTemplates[hintFormat] ?? "");

  // Streaming state for the active (in-progress) turn.
  const [streamState,   setStreamState]   = useState<StreamState>("idle");
  const [streamingText, setStreamingText] = useState("");
  const [streamError,   setStreamError]   = useState<string | null>(null);
  const [latencyMs,     setLatencyMs]     = useState<number | null>(null);
  const [promptTok,     setPromptTok]     = useState<number | null>(null);
  const [completionTok, setCompletionTok] = useState<number | null>(null);

  // Per-turn save state (keyed by turn index).
  const [turnSaveStates, setTurnSaveStates] = useState<Record<number, TurnSaveState>>({});

  const runIdRef      = useRef("");
  const cursorRef     = useRef(0);
  const chunkCountRef = useRef(0);
  const prevSampleKey = useRef("");
  const scrollRef     = useRef<HTMLDivElement>(null);
  const textareaRef   = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { ensureStyles(); }, []);

  // ── Sync sample from Python ───────────────────────────────────────────────
  useEffect(() => {
    const newPath = data?.filepath   ?? "";
    const newId   = data?.sample_id  ?? "";
    const newMt   = data?.media_type ?? "image";
    const newFr   = data?.frame_rate ?? null;
    if (!newId || !newPath) return;
    const sampleKey = JSON.stringify([newId, newPath, newMt, newFr]);
    if (sampleKey === prevSampleKey.current) return;
    prevSampleKey.current = sampleKey;

    log("sample synced", { sample_id: newId, filepath: newPath, media_type: newMt, frame_rate: newFr });

    setFilepath(newPath);
    setSampleId(newId);
    setMediaType(newMt);
    setFrameRate(newFr);
    setQuestion("");
    setStreamingText("");
    setStreamState("idle");
    setStreamError(null);
    setLatencyMs(null);
    setPromptTok(null);
    setCompletionTok(null);
    setTurnSaveStates({});
    setHintEditing(false);
    setHintTexts({});
    setMaxTokens(DEFAULT_MAX_TOKENS);
    setTemperature(DEFAULT_TEMPERATURE);
    setTopP(DEFAULT_TOP_P);
    setTopK(DEFAULT_TOP_K);
    runIdRef.current  = "";
    cursorRef.current = 0;
    chunkCountRef.current = 0;

    const cached = loadSession(newId);
    if (cached) {
      log("restored session", { sample_id: newId, turns: cached.turns.length });
      setTurns(cached.turns);
      setThinking(cached.thinking ?? DEFAULT_THINKING);
      setHintFormat(cached.hintFormat ?? "auto");
      setNFrames(cached.nFrames ?? 8);
      setImageMaxSide(cached.imageMaxSide ?? DEFAULT_IMAGE_MAX_SIDE);
      setHintTexts(cached.hintTexts ?? {});
      setMaxTokens(cached.maxTokens ?? DEFAULT_MAX_TOKENS);
      setTemperature(cached.temperature ?? DEFAULT_TEMPERATURE);
      setTopP(cached.topP ?? DEFAULT_TOP_P);
      setTopK(cached.topK ?? DEFAULT_TOP_K);
    } else {
      setTurns([]);
    }
  }, [data?.filepath, data?.sample_id, data?.media_type, data?.frame_rate]);

  // ── Persist turns ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!sampleId) return;
    saveSession(sampleId, {
      turns, thinking, hintFormat, nFrames, imageMaxSide, hintTexts,
      maxTokens, temperature, topP, topK,
    });
  }, [turns, thinking, hintFormat, nFrames, imageMaxSide, hintTexts,
      maxTokens, temperature, topP, topK, sampleId]);

  // ── Stream polling ────────────────────────────────────────────────────────
  useEffect(() => {
    if (streamState !== "running") return;
    const id = setInterval(() => {
      getStreamChunk(runIdRef.current, cursorRef.current)
        .then((chunk) => {
          if (chunk.text) {
            setStreamingText((prev) => prev + chunk.text);
            cursorRef.current = chunk.cursor;
          }
          chunkCountRef.current++;
          if (chunkCountRef.current % 10 === 0) {
            log(`stream chunk #${chunkCountRef.current}`, { cursor: chunk.cursor, done: chunk.done });
          }

          if (chunk.done) {
            clearInterval(id);
            const fs = chunk.final_status;
            if (fs?.status === "error") {
              logError("stream error", fs.error);
              setStreamError(fs.error ?? "Inference failed.");
              setStreamState("error");
            } else {
              log("stream done", {
                run_id:            runIdRef.current,
                latency_ms:        fs?.latency_ms,
                prompt_tokens:     fs?.prompt_tokens,
                completion_tokens: fs?.completion_tokens,
                detected_format:   fs?.detected_format ?? null,
              });
              setLatencyMs(fs?.latency_ms ?? null);
              setPromptTok(fs?.prompt_tokens ?? null);
              setCompletionTok(fs?.completion_tokens ?? null);
              setStreamState("done");

              const detectedFmt  = fs?.detected_format ?? null;
              const defaultField = fs?.default_field ?? "";
              const committedRunId = runIdRef.current;
              setStreamingText((text) => {
                setTurns((prev) => {
                  const last = prev[prev.length - 1];
                  if (!last || last.role !== "assistant") {
                    return [...prev, {
                      role:            "assistant",
                      content:         text,
                      run_id:          committedRunId,
                      detected_format: detectedFmt,
                      default_field:   defaultField,
                    }];
                  }
                  return prev;
                });
                return "";
              });
            }
          }
        })
        .catch((err) => logError("getStreamChunk error", err));
    }, 250);
    return () => clearInterval(id);
  }, [getStreamChunk, streamState]);

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, streamingText]);

  // ── Auto-grow composer ────────────────────────────────────────────────────
  // Reflow the textarea to fit its content (single line → multi-line), capped
  // so a long question scrolls internally rather than swallowing the panel.
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 140)}px`;
  }, [question]);

  // ── Send ──────────────────────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    if (!question.trim() || streamState === "running" || !filepath) return;

    const q = question.trim();
    setQuestion("");
    setStreamingText("");
    setStreamState("running");
    setStreamError(null);
    setLatencyMs(null);
    setPromptTok(null);
    setCompletionTok(null);
    cursorRef.current = 0;
    chunkCountRef.current = 0;

    const historyForApi = [...turns];
    setTurns((prev) => [...prev, { role: "user", content: q }]);

    // currentHintText is already "" when hintFormat is "auto".
    log("ask →", { filepath, media_type: mediaType, history_len: historyForApi.length,
                    thinking, hint_format: hintFormat,
                    hint_text_len: currentHintText.length, n_frames: nFrames,
                    image_max_side: imageMaxSide, max_tokens: maxTokens,
                    temperature, top_p: topP, top_k: topK });

    try {
      const result = await ask({
        filepath,
        media_type:      mediaType,
        question:        q,
        history:         historyForApi,
        thinking,
        hint_format:     hintFormat,
        hint_text:       currentHintText,
        n_frames:        nFrames,
        image_max_side:  imageMaxSide,
        max_tokens:      maxTokens,
        temperature,
        top_p:           topP,
        top_k:           topK,
      });
      log("ask ← run_id:", result.run_id);
      runIdRef.current = result.run_id;
    } catch (e: unknown) {
      const message = errorMessage(e, "Request failed.");
      logError("ask failed", message);
      setStreamError(message);
      setStreamState("error");
    }
  }, [question, streamState, filepath, mediaType, turns, thinking, hintFormat, currentHintText, nFrames, imageMaxSide, maxTokens, temperature, topP, topK, ask]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    },
    [handleSend]
  );

  // ── Convert to FiftyOne ───────────────────────────────────────────────────
  const handleConvert = useCallback(async (turnIdx: number, turn: Turn) => {
    const state = turnSaveStates[turnIdx];
    const fieldName = (state?.fieldName ?? turn.default_field ?? "").trim();
    if (!fieldName || !turn.run_id || !turn.detected_format || !sampleId) return;

    log("convert →", {
      turn_idx:        turnIdx,
      run_id:          turn.run_id,
      detected_format: turn.detected_format,
      field_name:      fieldName,
      sample_id:       sampleId,
      frame_rate:      frameRate,
    });

    setTurnSaveStates((prev) => ({
      ...prev,
      [turnIdx]: { ...prev[turnIdx], fieldName, saving: true, saved: false, error: null },
    }));

    try {
      // The save_minimax_label operator writes the field and returns the saved
      // label as sample-JSON. We then refresh the open modal in place via
      // useRefreshSample (below) so the overlay appears without a full-page
      // reload; a brand-new field is registered in the schema by the operator's
      // reload_dataset() call.
      const result = await new Promise<SaveLabelResult>((resolve, reject) => {
        saveLabelExecutor.execute(
          {
            run_id:          turn.run_id,
            sample_id:       sampleId,
            field_name:      fieldName,
            detected_format: turn.detected_format,
            frame_rate:      frameRate,
          },
          {
            callback: (res) => {
              if (res.error) {
                reject(new Error(errorMessage(res.error, "Save failed.")));
                return;
              }
              const r = res.result as (SaveLabelResult & { error?: string }) | null;
              if (!r) { reject(new Error("Save failed: operator returned no result.")); return; }
              if (r.error) { reject(new Error(r.error)); return; }
              resolve(r);
            },
          }
        );
      });
      log("convert ←", { label_type: result.label_type, count: result.count, field: result.field });

      // Refresh the open modal in place: merge the saved field into the cached
      // sample and bump the App's refresher. This shows the overlay without
      // closing the modal, so prev/next navigation survives.
      const current = modalSample?.sample;
      if (result.label_json && current && current._id === sampleId) {
        refreshSample({ ...current, [result.field]: result.label_json });
        log("convert refresh", { field: result.field, sample_id: sampleId });
      } else {
        logError("convert refresh skipped", {
          has_label_json: Boolean(result.label_json),
          has_sample:     Boolean(current),
          sample_match:   current?._id === sampleId,
        });
      }

      setTurnSaveStates((prev) => ({
        ...prev,
        [turnIdx]: {
          ...prev[turnIdx],
          saving:         false,
          saved:          true,
          savedLabelType: result.label_type,
          savedCount:     result.count,
          error:          null,
        },
      }));
    } catch (e: unknown) {
      const message = errorMessage(e, "Save failed.");
      logError("convert failed", message);
      setTurnSaveStates((prev) => ({
        ...prev,
        [turnIdx]: { ...prev[turnIdx], saving: false, saved: false, error: message },
      }));
    }
  }, [turnSaveStates, sampleId, frameRate, saveLabelExecutor, modalSample, refreshSample]);

  // ── API key warning ───────────────────────────────────────────────────────
  if (data?.api_key_missing) {
    return (
      <div style={{ padding: "20px 24px", fontFamily: V.font, color: V.red,
                    background: V.redBg, borderRadius: 6, margin: 16,
                    border: "1px solid #7a3030", lineHeight: 1.6 }}>
        <div style={{ fontWeight: 700, marginBottom: 8 }}>⚠ HF_TOKEN is not set</div>
        <div style={{ fontSize: 12, color: "#f5c6c6" }}>
          Export a Hugging Face token (with Inference Providers access) before launching FiftyOne,
          then restart the server.
        </div>
        <code style={{ display: "block", marginTop: 10, background: "#1a1a1a",
                       color: "#90cdf4", fontFamily: "monospace", fontSize: 11,
                       padding: "6px 10px", borderRadius: 4 }}>
          export HF_TOKEN="hf_...."
        </code>
      </div>
    );
  }

  if (!filepath) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
                    height: "100%", color: V.muted, fontSize: 13, fontFamily: V.font,
                    textAlign: "center", padding: 24 }}>
        Open a sample to start chatting.
      </div>
    );
  }

  const canSend    = !!question.trim() && streamState !== "running";
  const mediaLabel = mediaType === "video" ? "video" : "image";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%",
                  overflow: "hidden", fontFamily: V.font, fontSize: 13,
                  color: V.text, background: V.bg, boxSizing: "border-box" }}>

      {/* ── Sample info bar ── */}
      <div style={{ flexShrink: 0, padding: "6px 12px",
                    borderBottom: `1px solid ${V.divider}`,
                    display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 10,
                       background: V.bg3, color: V.muted, textTransform: "uppercase",
                       letterSpacing: "0.05em", flexShrink: 0 }}>
          {mediaLabel}
        </span>
        <span style={{ fontSize: 11, color: V.dim, overflow: "hidden",
                       textOverflow: "ellipsis", whiteSpace: "nowrap" }}
              title={filepath}>
          {filepath.split("/").pop()}
        </span>
      </div>

      {/* ── Chat history ── */}
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto",
                                     padding: "12px 14px", minHeight: 0 }}>

        {turns.length === 0 && streamState === "idle" && (
          <div style={{ color: V.muted, fontSize: 12, textAlign: "center",
                        padding: "24px 12px" }}>
            Ask anything about this {mediaLabel}…
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} style={{ marginBottom: 10 }}>
            <div style={{
              display:        "flex",
              justifyContent: turn.role === "user" ? "flex-end" : "flex-start",
            }}>
              <div style={{
                maxWidth:     "88%",
                background:   turn.role === "user" ? V.primary : V.bg2,
                color:        turn.role === "user" ? V.textInv  : V.text,
                borderRadius: turn.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
                padding:      "8px 12px",
                fontSize:     13,
                lineHeight:   1.5,
              }}>
                {turn.role === "assistant" ? (
                  <div className="mmx-md">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                  </div>
                ) : (
                  turn.content
                )}
              </div>
            </div>

            {turn.role === "assistant" && turn.detected_format && (
              <ConvertBar
                turnIdx={i}
                turn={turn}
                saveState={turnSaveStates[i]}
                onConvert={handleConvert}
                onFieldChange={(idx, name) =>
                  setTurnSaveStates((prev) => ({
                    ...prev,
                    [idx]: { ...prev[idx], fieldName: name, saved: false, error: null,
                             saving: false, savedLabelType: undefined, savedCount: undefined },
                  }))
                }
              />
            )}
          </div>
        ))}

        {streamState === "running" && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
            <div style={{
              maxWidth: "88%", background: V.bg2, color: V.text,
              borderRadius: "12px 12px 12px 4px", padding: "8px 12px",
              fontSize: 13, lineHeight: 1.5,
            }}>
              {streamingText ? (
                <div className="mmx-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingText}</ReactMarkdown>
                  <span style={{
                    display: "inline-block", width: 2, height: "1em",
                    background: V.primary, marginLeft: 1, verticalAlign: "text-bottom",
                    animation: "mmxBlink 1s step-end infinite",
                  }} />
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: 6, color: V.muted }}>
                  <span style={{
                    width: 12, height: 12, borderRadius: "50%",
                    border: `2px solid ${V.divider}`, borderTop: `2px solid ${V.primary}`,
                    animation: "mmxSpin 0.7s linear infinite",
                    display: "inline-block", flexShrink: 0,
                  }} />
                  Waiting for response…
                </div>
              )}
            </div>
          </div>
        )}

        {streamError && (
          <div style={{ padding: "8px 10px", background: V.redBg, color: V.red,
                        borderRadius: 6, fontSize: 12, marginBottom: 8 }}>
            {streamError}
          </div>
        )}

        {streamState === "done" && (latencyMs != null || completionTok != null) && (
          <div style={{ fontSize: 11, color: V.dim, textAlign: "center", padding: "2px 0 8px" }}>
            {[
              completionTok != null && `${completionTok} tokens`,
              latencyMs    != null && `${(latencyMs / 1000).toFixed(1)}s`,
              promptTok    != null && `${promptTok} in`,
            ].filter(Boolean).join(" · ")}
          </div>
        )}
      </div>

      {/* ── Bottom bar ── */}
      <div style={{ flexShrink: 0, padding: "8px 12px",
                    borderTop: `1px solid ${V.divider}`,
                    background: V.bg2, display: "flex",
                    flexDirection: "column", gap: 7 }}>

        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" as const }}>

          {/* Output format selector */}
          <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: V.muted }}>
            <span style={{ flexShrink: 0 }}>Output:</span>
            <select
              value={hintFormat}
              onChange={(e) => { setHintFormat(e.target.value); setHintEditing(false); }}
              style={{
                background: V.bg2, color: V.text,
                border: `1px solid ${V.divider}`, borderRadius: 4,
                padding: "2px 5px", fontSize: 11, fontFamily: V.font,
                cursor: "pointer", outline: "none",
              }}
            >
              <option value="auto">Auto</option>
              <option value="box">Boxes</option>
              <option value="point">Keypoints</option>
              <option value="temporal">Temporal</option>
            </select>
          </div>

          {/* Frames-to-sample (video only) */}
          {mediaType === "video" && (
            <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: V.muted }}>
              <span style={{ flexShrink: 0 }}>Frames:</span>
              <input
                type="number" min={1} max={16} value={nFrames}
                onChange={(e) => setNFrames(Math.max(1, Math.min(16, Number(e.target.value) || 8)))}
                style={{
                  width: 44, background: V.bg2, color: V.text,
                  border: `1px solid ${V.divider}`, borderRadius: 4,
                  padding: "2px 5px", fontSize: 11, fontFamily: V.font, outline: "none",
                }}
              />
            </div>
          )}

          {/* Advanced settings toggle (reasoning, detail, sampling) */}
          <button
            onClick={() => setShowAdvanced((v) => !v)}
            title="Advanced settings (reasoning, detail, sampling)"
            style={{
              marginLeft: "auto", background: "none", border: "none",
              cursor: "pointer", color: showAdvanced ? V.primary : V.muted,
              fontSize: 11, fontFamily: V.font, display: "flex",
              alignItems: "center", gap: 4, padding: 0, flexShrink: 0,
            }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.488.488 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z" />
            </svg>
            Advanced
          </button>
        </div>

        {/* ── Format-instruction preview / editor (kept next to Output) ── */}
        {hintFormat === "auto" ? (
          <div style={{ fontSize: 10, color: V.dim, fontStyle: "italic", padding: "0 2px" }}>
            Auto — no format instruction is added; the model decides the output shape.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 5,
                        background: V.bg, border: `1px solid ${V.divider}`,
                        borderRadius: 6, padding: "6px 8px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
              <span style={{ fontSize: 10, color: V.dim, textTransform: "uppercase",
                             letterSpacing: "0.04em" }}>
                Instruction appended to your question
              </span>
              <button
                onClick={() => setHintEditing((v) => !v)}
                title={hintEditing ? "Finish editing" : "Edit the format instruction"}
                style={{ background: "none", border: "none", cursor: "pointer", color: V.primary,
                         fontSize: 11, display: "flex", alignItems: "center", gap: 3,
                         padding: 0, fontFamily: V.font, flexShrink: 0 }}
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 000-1.41l-2.34-2.34a1 1 0 00-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z" />
                </svg>
                {hintEditing ? "Done" : "Edit"}
              </button>
            </div>

            {hintEditing ? (
              <>
                <textarea
                  value={currentHintText}
                  onChange={(e) =>
                    setHintTexts((prev) => ({ ...prev, [hintFormat]: e.target.value }))
                  }
                  rows={3}
                  spellCheck={false}
                  style={{
                    width: "100%", background: V.bg2, color: V.text,
                    border: `1px solid ${V.divider}`, borderRadius: 4,
                    padding: "6px 8px", fontSize: 11, fontFamily: "ui-monospace, monospace",
                    lineHeight: 1.5, outline: "none", resize: "vertical" as const,
                    boxSizing: "border-box" as const,
                  }}
                />
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <button
                    onClick={() =>
                      setHintTexts((prev) => ({
                        ...prev,
                        [hintFormat]: hintTemplates[hintFormat] ?? "",
                      }))
                    }
                    style={{ background: "none", border: "none", cursor: "pointer",
                             color: V.dim, fontSize: 10, textDecoration: "underline",
                             padding: 0, fontFamily: V.font }}
                  >
                    Reset to default
                  </button>
                </div>
              </>
            ) : (
              <code style={{ fontSize: 11, color: V.muted, fontFamily: "ui-monospace, monospace",
                             lineHeight: 1.5, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {currentHintText}
              </code>
            )}
          </div>
        )}

        {/* ── Advanced settings (reasoning, detail, sampling) ── */}
        {showAdvanced && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8,
                        background: V.bg, border: `1px solid ${V.divider}`,
                        borderRadius: 6, padding: "8px 10px" }}>
            <RadioGroup<string>
              label="Thinking" name="mmx-thinking" value={thinking}
              title="M3 reasoning. Off is fastest and cleanest for JSON; Adaptive lets the model decide; Always forces reasoning."
              options={THINKING_MODES} onChange={setThinking}
            />
            {mediaType === "image" && (
              <RadioGroup<number>
                label="Detail" name="mmx-detail" value={imageMaxSide}
                title="Resolution sent to M3 (longest side). Higher helps small objects; Native sends full resolution."
                options={IMAGE_RESOLUTIONS}
                onChange={setImageMaxSide}
              />
            )}
            <div style={{ height: 1, background: V.divider, margin: "1px 0" }} />
            <div style={{ display: "grid",
                          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                          gap: "7px 16px" }}>
              <GenParam label="Max tokens" title="Output token ceiling."
                value={maxTokens} min={1} step={64}
                onChange={(n) => setMaxTokens(Math.max(1, Math.round(n)))} width={64} />
              <GenParam label="Temp" title="Sampling temperature (M3 default 1.0)."
                value={temperature} min={0} max={2} step={0.05}
                onChange={(n) => setTemperature(Math.max(0, Math.min(2, n)))} width={64} />
              <GenParam label="Top-p" title="Nucleus-sampling mass (M3 default 0.95)."
                value={topP} min={0} max={1} step={0.05}
                onChange={(n) => setTopP(Math.max(0, Math.min(1, n)))} width={64} />
              <GenParam label="Top-k" title="Top-k sampling (M3 default 40; some providers ignore it)."
                value={topK} min={1} step={1}
                onChange={(n) => setTopK(Math.max(1, Math.round(n)))} width={64} />
            </div>
            <button
              onClick={() => {
                setThinking(DEFAULT_THINKING);
                setImageMaxSide(DEFAULT_IMAGE_MAX_SIDE);
                setMaxTokens(DEFAULT_MAX_TOKENS);
                setTemperature(DEFAULT_TEMPERATURE);
                setTopP(DEFAULT_TOP_P);
                setTopK(DEFAULT_TOP_K);
              }}
              style={{ alignSelf: "flex-end", background: "none", border: "none",
                       cursor: "pointer", color: V.dim, fontSize: 10,
                       textDecoration: "underline", padding: 0, fontFamily: V.font }}
            >
              Reset to defaults
            </button>
          </div>
        )}

        <div
          style={{
            display: "flex", alignItems: "flex-end", gap: 6,
            background: V.bg,
            border: `1px solid ${inputFocused ? V.primary : V.divider}`,
            borderRadius: 14,
            padding: "5px 5px 5px 12px",
            boxShadow: inputFocused
              ? `0 0 0 3px color-mix(in srgb, ${V.primary} 22%, transparent)`
              : "none",
            transition: "border-color 120ms ease, box-shadow 120ms ease",
            boxSizing: "border-box" as const,
          }}
        >
          <textarea
            ref={textareaRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
            disabled={streamState === "running"}
            placeholder={
              streamState === "running"
                ? "Waiting for response…"
                : `Ask about this ${mediaLabel}… (Enter ↵ to send)`
            }
            rows={1}
            style={{
              flex: 1, background: "transparent",
              color: streamState === "running" ? V.dim : V.text,
              border: "none", outline: "none", resize: "none" as const,
              padding: "5px 0", margin: 0,
              fontSize: 13, fontFamily: V.font, lineHeight: 1.5,
              maxHeight: 140, overflowY: "auto",
              boxSizing: "border-box" as const,
              cursor: streamState === "running" ? "not-allowed" : "text",
            }}
          />
          <button
            onClick={handleSend} disabled={!canSend}
            title="Send (Enter)"
            style={{
              width: 30, height: 30, borderRadius: "50%", flexShrink: 0,
              border: "none", padding: 0,
              background: canSend ? V.primary : "transparent",
              color:      canSend ? V.textInv : V.dim,
              cursor:     canSend ? "pointer" : "default",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "background 120ms ease, color 120ms ease",
            }}
          >
            {streamState === "running" ? (
              <span style={{
                width: 16, height: 16, borderRadius: "50%",
                border: `2px solid ${V.divider}`, borderTop: `2px solid ${V.primary}`,
                animation: "mmxSpin 0.7s linear infinite", display: "inline-block",
              }} />
            ) : (
              <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            )}
          </button>
        </div>

        {turns.length > 0 && streamState !== "running" && (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              onClick={() => {
                setTurns([]);
                setStreamingText("");
                setStreamState("idle");
                setStreamError(null);
                setTurnSaveStates({});
                log("conversation cleared for sample", sampleId);
              }}
              style={{
                background: "none", border: "none", cursor: "pointer",
                color: V.dim, fontSize: 11, padding: "0 2px",
                textDecoration: "underline", fontFamily: V.font,
              }}
            >
              Clear conversation
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// ConvertBar
// ---------------------------------------------------------------------------

interface ConvertBarProps {
  turnIdx:       number;
  turn:          Turn;
  saveState:     TurnSaveState | undefined;
  onConvert:     (idx: number, turn: Turn) => void;
  onFieldChange: (idx: number, name: string) => void;
}

const ConvertBar: React.FC<ConvertBarProps> = ({
  turnIdx, turn, saveState, onConvert, onFieldChange,
}) => {
  const fieldName  = saveState?.fieldName ?? turn.default_field ?? "";
  const saving     = saveState?.saving    ?? false;
  const saved      = saveState?.saved     ?? false;
  const saveError  = saveState?.error     ?? null;
  const fmtLabel   = FORMAT_LABELS[turn.detected_format ?? ""] ?? turn.detected_format;
  const canConvert = !!fieldName.trim() && !saving && !saved;

  return (
    <div style={{
      marginTop:    6,
      marginLeft:   0,
      padding:      "7px 10px",
      background:   V.bg3,
      borderRadius: "0 0 8px 8px",
      border:       `1px solid ${V.divider}`,
      borderTop:    "none",
      display:      "flex",
      flexDirection: "column" as const,
      gap:          5,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: V.muted }}>
        <span style={{
          fontSize: 10, padding: "1px 6px", borderRadius: 8,
          background: V.bg2, color: V.primary,
          border: `1px solid ${V.divider}`,
        }}>
          {fmtLabel}
        </span>
        <span>detected — save to dataset?</span>
      </div>

      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <input
          type="text"
          value={fieldName}
          onChange={(e) => onFieldChange(turnIdx, e.target.value)}
          disabled={saving || saved}
          placeholder="field name"
          style={{
            flex:         1,
            background:   (saving || saved) ? V.bg : V.bg2,
            color:        (saving || saved) ? V.muted : V.text,
            border:       `1px solid ${V.divider}`,
            borderRadius: 4,
            padding:      "4px 8px",
            fontSize:     11,
            fontFamily:   "ui-monospace, monospace",
            outline:      "none",
            cursor:       (saving || saved) ? "not-allowed" : "text",
          }}
        />
        <button
          onClick={() => onConvert(turnIdx, { ...turn, default_field: fieldName })}
          disabled={!canConvert}
          title={saved ? "Already saved" : `Save as ${fmtLabel} to field "${fieldName}"`}
          style={{
            background:   saved  ? V.greenBg  : canConvert ? V.bg2 : V.bg,
            color:        saved  ? V.green     : canConvert ? V.text : V.dim,
            border:       `1px solid ${saved ? V.greenBorder : V.divider}`,
            borderRadius: 4,
            padding:      "4px 10px",
            fontSize:     11,
            fontFamily:   V.font,
            cursor:       canConvert ? "pointer" : "not-allowed",
            flexShrink:   0,
            whiteSpace:   "nowrap" as const,
            display:      "flex",
            alignItems:   "center",
            gap:          4,
          }}
        >
          {saving ? (
            <>
              <span style={{
                width: 10, height: 10, borderRadius: "50%",
                border: `2px solid ${V.divider}`, borderTop: `2px solid ${V.primary}`,
                animation: "mmxSpin 0.7s linear infinite", display: "inline-block",
              }} />
              Saving…
            </>
          ) : saved ? (
            `✓ ${saveState?.savedCount ?? ""} ${saveState?.savedLabelType ?? ""}`
          ) : (
            "Convert to FiftyOne ▶"
          )}
        </button>
      </div>

      {saveError && (
        <div style={{ fontSize: 11, color: V.red, padding: "2px 0" }}>
          {saveError}
        </div>
      )}
    </div>
  );
};

export default MiniMaxChatPanel;
