import gradio as gr
import json

# --- STATIC TEMPLATE DATA (JSON-like dict) ---
TEMPLATE_DATA = {
    "Redefining Success": {
        "title": "Redefining Success",
        "structure": (
            "Only [a small fraction] of [specific initiatives/individuals] [achieve a desirable outcome].\n"
            "But you don’t have to follow the [typical definition of success].\n"
            "Define what success means to you - something you’ll look back on in [time period] with pride.\n"
            "Maybe it’s\n"
            "Or perhaps it’s\n"
            "Or even\n"
            "No matter the path, the choice is yours to make."
        ),
        "example": (
            "Only 10% of startups secure venture capital funding in their first year.\n"
            "But you don’t have to follow the typical definition of success.\n"
            "Define what success means to you - something you’ll look back on in 5 years with pride.\n"
            "Maybe it’s\n"
            "Or perhaps it’s\n"
            "Or even\n"
            "No matter the path, the choice is yours to make."
        )
    },
    "Promises vs Reality": {
        "title": "Promises vs Reality",
        "structure": (
            "What [Technology/Product] promised: [Grand promise].\n"
            "What [Technology/Product] delivered: [Funny, mundane, or ironic reality]."
        ),
        "example": (
            "What AI promised: End of manual work.\n"
            "What AI delivered: A second layer of manual work to double-check the AI."
        )
    },
    "Turning Point in Life": {
        "title": "Turning Point in Life",
        "structure": (
            "You hit [specific age or milestone], and suddenly [group of people or community] "
            "starts [unexpected or stereotypical activity]."
        ),
        "example": (
            "You turn 30 and the whole squad starts playing pickleball or running half marathons."
        )
    },
    "Finding Motivation": {  # I interpreted the title from your example content
        "title": "Finding Motivation",
        "structure": (
            "Finding motivation can be tough, especially when [specific challenge] feels overwhelming.\n"
            "Here are a few things that keep me energized and focused: [Practical habit or routine].\n"
            "What’s your favorite way to push through tough days?"
        ),
        "example": (
            "Finding motivation can be tough, especially when you’re a creator battling a creative block.\n"
            "Here are a few things that helped me get back on track:\n"
            "1. Taking a step back to recharge and reflect - it’s okay to pause."
        )
    }
}


# --- List of template names for the dropdown ---
template_options = list(TEMPLATE_DATA.keys())

# --- Function to update the template info display (acts like a pop-up) ---
def update_template_info(template_name):
    data = TEMPLATE_DATA.get(template_name, {})
    return f"""### 📋 {data.get('title', template_name)}

**Structure**  
{data.get('structure', 'No structure defined.')}

**Example**  
{data.get('example', 'No example provided.')}
"""

# --- Generation function returning JSON ---
def generate_post(topic, word_count, language, tone, style, goal, target_audience, call_to_action, template_name):
    # Fetch full template data
    template_data = TEMPLATE_DATA.get(template_name, {})

    # 1. Bundle all inputs into a JSON-serializable dict
    inputs = {
        "topic": topic,
        "word_count": word_count,
        "language": language,
        "tone": tone,
        "style": style,
        "goal": goal,
        "target_audience": target_audience,
        "call_to_action": call_to_action,
        "template": {
            "name": template_name,
            "structure": template_data.get("structure", ""),
            "example": template_data.get("example", "")
        }
    }

    # 2. Placeholder for the actual generated post – replace with your AI logic
    generated_text = (
        f"**Generated Post (placeholder)**\n\n"
        f"Topic: {topic}\n"
        f"Using template: {template_name}\n\n"
        f"(Your AI-generated post would appear here – "
        f"connect to Lenny's Newsletter data to make it real.)"
    )

    # 3. Combine inputs and output into a single JSON payload
    result = {
        "inputs": inputs,
        "generated_post": generated_text
    }

    return json.dumps(result, indent=2)


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
with gr.Blocks(theme=gr.themes.Soft(), title="LinkedIn Post Generator") as demo:
    gr.Markdown("# ✍️ LinkedIn Post Generator")
    gr.Markdown("Create authentic, non‑generic posts inspired by Lenny’s Newsletter")

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
                value="Redefining Success"
            )
            # This Markdown will update instantly when the dropdown changes
            template_info = gr.Markdown(
                value=update_template_info("Redefining Success"),
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

            media = gr.File(
                label="Add Media",
                file_types=[".png", ".jpg", ".jpeg", ".gif", ".mp4", ".pdf"],
                file_count="multiple"
            )

            generate_btn = gr.Button("Generate Post", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="JSON Output (inputs + generated post)",
                lines=25,
                interactive=False,
                show_copy_button=True
            )

    # --- Event wiring ---
    # Update the info panel whenever the template dropdown changes
    template_dropdown.change(
        fn=update_template_info,
        inputs=template_dropdown,
        outputs=template_info
    )

    # Generate the JSON output
    generate_btn.click(
        fn=generate_post,
        inputs=[topic, word_count, language, tone, style, goal, target_audience, call_to_action, template_dropdown],
        outputs=output
    )

    gr.Markdown("---\n💡 *Select a template to see its structure and example appear instantly – like a built‑in preview pop‑up.*")

if __name__ == "__main__":
    demo.launch()