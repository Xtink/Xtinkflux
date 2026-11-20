import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_FLUX_CHARS = 30000
MAX_MESSAGE_CHARS = 12000
MAX_ATTACHMENTS = 8
MAX_IMAGE_CHARS = 7_000_000
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

SYSTEM_PROMPT = r"""
You are Xbot, the coding-only AI architect and native FLUX reasoning/build engine inside Xtink.
You ONLY assist with programming, FLUX language, software engineering, web/app development,
debugging, code review, technical explanations, project files, and related development tasks.
If a user asks about unrelated subjects (general trivia, personal advice, entertainment, etc.),
politely say you only help with coding and software development. Do not answer the unrelated request.

ABOUT FLUX
FLUX is a compact, context-dependent programming language. It is NOT ordinary prose,
and it is NOT a conventional language with a fixed dictionary of keywords. A word's
meaning comes from its combination with other words, its position, the surrounding
instructions, the current project, and the user's goal. Never assign a permanent,
universal meaning to an individual FLUX word. Do not require braces, commas, colons,
semicolons, or any particular syntax unless the user used them.

YOUR BUILD RESPONSIBILITY
When the request has action="build" (or the user explicitly asks you to run, build,
create, implement, or generate from the supplied FLUX), you MUST produce actual usable
project source files. Do not merely explain the FLUX, write a tutorial, propose files,
or tell the user to run an imaginary command such as `flux build` or `xtink run`.

Read the ENTIRE FLUX source as one connected instruction. Infer the intended result.
For a web page or interface, normally generate a complete `index.html` containing the
HTML, CSS, and JavaScript needed for the requested result. Add separate files only when
they materially help. Make the result visually complete, responsive, and usable.

Example interpretation principle (not a keyword dictionary): if a user writes a page
header instruction and describes a secondary heading as upside-down, blue, and bold
Roboto, implement that visual result in CSS. Do not explain that the words might mean
those things; create the page. If an instruction is slightly unusual but its intent is
reasonably inferable, implement the most sensible interpretation.

If the user gives a design instruction, preserve its exact visible text. Do not add
unrequested page copy, menus, sections, or features. For styling requests, implement
the style in real CSS. Use accessible semantic HTML and include responsive behavior.

PROJECT CONTEXT
Use any supplied project files and their contents to update the existing project
consistently. If an existing app is provided, modify the relevant files rather than
replacing unrelated work. Never include API keys, passwords, or secrets.

OUTPUT RULES
Return ONLY one valid JSON object:
{
  "reply": "brief description of what was built",
  "files": [
    {"path": "index.html", "content": "complete file contents"}
  ]
}
For a BUILD request, `files` MUST contain at least one complete file. Never return an
empty files array for a build request. Use safe relative paths only; never absolute
paths or `..`. Do not wrap JSON in markdown fences. Do not claim files were deployed,
saved to disk, or executed. XTINK will add returned files to its workspace.
For ordinary Xbot conversation that is not a build request, answer normally and use
an empty files array.
"""


def extract_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Xbot returned a response that was not valid JSON.")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Xbot response must be a JSON object.")
    return value


def clean_files(value: Any) -> list[dict[str, str]]:
    # Accept either the required [{path, content}, ...] shape or a simple
    # {"filename": "file contents"} mapping if the model varies its formatting.
    if isinstance(value, dict):
        value = [{"path": path, "content": content} for path, content in value.items()]
    if not isinstance(value, list):
        return []

    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("filename") or item.get("name") or "").strip().replace("\\", "/")
        content = item.get("content", item.get("code", item.get("contents", "")))
        if not path or not isinstance(content, str) or not content.strip():
            continue
        if path.startswith("/") or re.match(r"^[a-zA-Z]:", path):
            continue
        if any(part in ("", ".", "..") for part in path.split("/")):
            continue
        if path in seen:
            continue
        seen.add(path)
        files.append({"path": path, "content": content})
    return files


def ask_model(client: Groq, messages: list[dict[str, Any]], model: str | None = None) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model or GROQ_MODEL,
        temperature=0.15,
        max_completion_tokens=12000,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = completion.choices[0].message.content or "{}"
    return extract_json(raw)


@app.get("/")
def home():
    return jsonify({"service": "XTINK Xbot API", "status": "online", "endpoints": ["/health", "/api/chat"]})


@app.get("/health")
def health():
    return jsonify({"ok": True, "groq_configured": bool(os.getenv("GROQ_API_KEY")), "model": GROQ_MODEL})


def prepare_attachments(raw: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return safe text attachments and image data URLs for the model."""
    text_items: list[dict[str, str]] = []
    images: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return text_items, images
    for item in raw[:MAX_ATTACHMENTS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "attachment"))[:180]
        mime = str(item.get("type", "application/octet-stream"))[:100].lower()
        if mime.startswith("image/"):
            data_url = item.get("data_url", "")
            if isinstance(data_url, str) and data_url.startswith("data:image/") and len(data_url) <= MAX_IMAGE_CHARS:
                images.append({"name": name, "data_url": data_url})
        else:
            content = item.get("content", "")
            if isinstance(content, str) and content:
                text_items.append({"name": name, "content": content[:20000]})
    return text_items, images


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    flux = str(payload.get("flux", ""))[:MAX_FLUX_CHARS]
    active_file = str(payload.get("active_file", "main.flux"))[:200]
    action = str(payload.get("action", "")).lower()
    build_mode = action == "build" or (bool(flux.strip()) and bool(re.search(r"\b(build|generate|implement|create|run)\b", message, re.I)))

    if not message:
        return jsonify({"error": "Please enter a message for Xbot."}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": "That message is too long."}), 400
    raw_attachments = payload.get("attachments", [])
    if isinstance(raw_attachments, list) and len(raw_attachments) > MAX_ATTACHMENTS:
        return jsonify({"error": f"Attach no more than {MAX_ATTACHMENTS} files at once."}), 400

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY is missing. Add it to your Render environment variables."}), 503

    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    text_attachments, image_attachments = prepare_attachments(payload.get("attachments", []))
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    # Keep only small, well-formed history entries to avoid sending arbitrary payloads.
    safe_history = []
    for entry in history[-10:]:
        if isinstance(entry, dict) and entry.get("role") in ("user", "assistant"):
            content = entry.get("content")
            if isinstance(content, str):
                safe_history.append({"role": entry["role"], "content": content[:4000]})

    user_context = {
        "action": "build" if build_mode else "chat",
        "active_editor_file": active_file,
        "flux_source": flux,
        "user_request": message,
        "project_context": context,
        "attachments": text_attachments,
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(safe_history)
    user_text = json.dumps(user_context, ensure_ascii=False)
    if image_attachments:
        user_content: Any = [{"type": "text", "text": user_text + "\nPlease inspect the attached image(s) as coding/design references. Describe or implement only what is relevant to the user's coding request."}]
        for image_item in image_attachments:
            user_content.append({"type": "text", "text": "Attached image: " + image_item["name"]})
            user_content.append({"type": "image_url", "image_url": {"url": image_item["data_url"]}})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_text})

    try:
        client = Groq(api_key=api_key)
        result = ask_model(client, messages, model=VISION_MODEL if image_attachments else GROQ_MODEL)
        files = clean_files(result.get("files", result.get("generated_files", [])))

        # A build request that receives prose but no files gets one explicit repair
        # attempt. This prevents the old false-success message in the XTINK UI.
        if build_mode and not files:
            repair_messages = messages + [
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                {"role": "user", "content": (
                    "REPAIR: Your previous response did not include any usable project files. "
                    "This is a build request. Generate the actual complete implementation now. "
                    "For this web-page request, return at least one complete index.html file. "
                    "Put the full source code in files[0].content. Do not explain FLUX or provide "
                    "instructions instead of code. Preserve the user's requested visible text and design."
                )},
            ]
            result = ask_model(client, repair_messages, model=VISION_MODEL if image_attachments else GROQ_MODEL)
            files = clean_files(result.get("files", result.get("generated_files", [])))

        reply = result.get("reply", "")
        if not isinstance(reply, str):
            reply = str(reply)

        if build_mode and not files:
            app.logger.error("Xbot returned no project files after repair attempt")
            return jsonify({"error": "Xbot did not return usable project files. Try Run FLUX again; the backend has rejected the empty build instead of reporting a false success."}), 502

        return jsonify({"reply": reply or ("Project files generated." if files else "Ready."), "files": files})
    except Exception as exc:
        app.logger.exception("Xbot/Groq request failed")
        return jsonify({"error": f"Xbot could not complete the request. Check the Render logs. ({type(exc).__name__})"}), 502


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
