import gradio as gr
import json
import logging
from pathlib import Path

from llm_integration import generate_post_from_inputs as run_generation_pipeline, LLMIntegrationError

logger = logging.getLogger(__name__)

# --- TEMPLATE DATA, loaded from /templates (single source of truth, also
# used by prompt_templates.py) ---
_TEMPLATE_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "prompt_templates.json"
)

try:
    with open(_TEMPLATE_DATA_PATH, "r", encoding="utf-8") as f:
        TEMPLATE_DATA = json.load(f)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Template file not found at {_TEMPLATE_DATA_PATH}. The app can't "
        "start without it."
    ) from exc
except json.JSONDecodeError as exc:
    raise ValueError(f"Template file at {_TEMPLATE_DATA_PATH} is not valid JSON: {exc}") from exc

if not TEMPLATE_DATA:
    raise ValueError(f"Template file at {_TEMPLATE_DATA_PATH} is empty.")


# --- List of template names for the dropdown ---
template_options = list(TEMPLATE_DATA.keys())
_default_template = template_options[0] if template_options else None


# --- Function to update the template info display (acts like a pop-up) ---
def update_template_info(template_name):
    data = TEMPLATE_DATA.get(template_name, {})
    return f"""### 📋 {data.get('title', template_name)}

**Structure**
{data.get('structure', 'No structure defined.')}

**Example**
{data.get('example', 'No example provided.')}
"""

# --- Generation function: runs the real filter -> draft -> polish pipeline ---
def generate_post_from_inputs(topic, word_count, language, tone, style, goal, target_audience, call_to_action, template_name):
    if not topic or not topic.strip():
        return "Please enter a topic before generating a post."

    inputs = {
        "topic": topic,
        "word_count": word_count,
        "language": language,
        "tone": tone,
        "style": style,
        "goal": goal,
        "target_audience": target_audience,
        "call_to_action": call_to_action,
        "template": template_name,
    }

    # Delegates to llm_integration.generate_post_from_inputs(), which
    # filters the knowledge base for relevant source articles, drafts the
    # post, and runs it through the editing/polish pass before returning
    # the final, ready-to-paste text. Any failure is caught here so the
    # Gradio app shows a readable message instead of crashing.
    try:
        return run_generation_pipeline(inputs)
    except LLMIntegrationError as exc:
        logger.error("Post generation failed: %s", exc)
        return f"Something went wrong generating this post: {exc}"
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        logger.exception("Unexpected error during post generation")
        return f"Unexpected error: {exc}"


# --- Dropdown choices (unchanged) ---
word_count_options = ["less than 50", "50-150", "150-500", "500+"]
language_options = ["English", "German", "Arabic"]

tone_options = [
    "Professional", "Casual", "Inspirational", "Humorous",
    "Authoritative", "Empathetic", "Motivational", "Thoughtful"
]
style_options = [
    "Storytelling", "Educational", "Thought Leadership",
    "Listicle", "How‑to", "Case Study", "Opinion"
]
goal_options = [
    "Build Personal Brand", "Engage Audience", "Drive Traffic",
    "Generate Leads", "Share Knowledge", "Inspire Action"
]
target_audience_options = [
    "Executives", "Founders", "Marketers", "Product Managers",
    "Engineers", "General Professionals", "Students"
]
call_to_action_options = [
    "Share Your Thoughts", "Comment Below", "Like & Repost",
    "Visit My Website", "Subscribe to Newsletter", "Download Resource"
]

# --- Gradio interface ---
# Note: theme is passed to launch() (not the Blocks constructor) and
# Textbox uses buttons=["copy"] instead of show_copy_button — both are
# Gradio 6 API changes.
with gr.Blocks(title="LinkedIn Post Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✍️ LinkedIn Post Generator")
    gr.Markdown("Create authentic, non‑generic posts inspired by Lenny’s Newsletter and Podcasts")

    with gr.Row():
        with gr.Column(scale=2):
            topic = gr.Textbox(
                label="Topic for the post",
                placeholder="Write what's on your mind...",
                lines=3
            )
            word_count = gr.Dropdown(
                choices=word_count_options,
                label="Word Count",
                value="150-500"
            )
            language = gr.Dropdown(
                choices=language_options,
                label="Language",
                value="English"
            )

            # --- TEMPLATE DROPDOWN + INFO PANEL (acts like a pop-up) ---
            template_dropdown = gr.Dropdown(
                choices=template_options,
                label="Template",
                value=_default_template
            )
            # This Markdown will update instantly when the dropdown changes
            template_info = gr.Markdown(
                value=update_template_info(_default_template) if _default_template else "No templates available.",
                label="Template Preview"
            )

            with gr.Accordion("Advanced Options", open=False):
                tone = gr.Dropdown(choices=tone_options, label="Tone", value="Professional")
                style = gr.Dropdown(choices=style_options, label="Style", value="Storytelling")
                goal = gr.Dropdown(choices=goal_options, label="Goal", value="Build Personal Brand")
                target_audience = gr.Dropdown(
                    choices=target_audience_options,
                    label="Target Audience",
                    value="Executives"
                )
                call_to_action = gr.Dropdown(
                    choices=call_to_action_options,
                    label="Call to Action",
                    value="Share Your Thoughts"
                )

            generate_btn = gr.Button("Generate Post", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Generated LinkedIn Post",
                lines=25,
                interactive=False,
                show_copy_button=True
                # buttons=["copy"]
            )

    # --- Event wiring ---
    # Update the info panel whenever the template dropdown changes
    template_dropdown.change(
        fn=update_template_info,
        inputs=template_dropdown,
        outputs=template_info
    )

    # Run the full generation pipeline and display the finished post
    generate_btn.click(
        fn=generate_post_from_inputs,
        inputs=[topic, word_count, language, tone, style, goal, target_audience, call_to_action, template_dropdown],
        outputs=output
    )

    gr.Markdown("---\n💡 *Select a template to see its structure and example appear instantly – like a built‑in preview pop‑up.*")

if __name__ == "__main__":
    demo.launch(share=True)
