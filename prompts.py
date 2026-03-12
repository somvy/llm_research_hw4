INPUT_POOL = [
    # IOI-style (indirect object identification)
    "When Mary and John went to the store, Mary gave a bottle of milk to",
    "When Alice and Bob went to the park, Alice handed the ball to",
    "After Tom and Sarah finished dinner, Tom passed the check to",
    "When Emma and James arrived at the party, Emma gave a present to",
    "After Lisa and Mike went shopping, Lisa handed the bags to",
    "When David and Rachel entered the room, David gave the keys to",
    "After Karen and Steve left the meeting, Karen sent an email to",
    "When Paul and Linda visited the museum, Paul showed the map to",
    "After Anna and Chris finished the project, Anna sent the report to",
    "When Julia and Mark went to lunch, Julia passed the menu to",
    "When Sam and Alex went hiking, Sam gave the water bottle to",
    "After Kate and Dan finished class, Kate lent the textbook to",
    "When Ben and Olivia arrived home, Ben handed the groceries to",
    "After Sophia and Ryan cooked dinner, Sophia served the food to",
    "When Noah and Ella went to the beach, Noah gave the sunscreen to",
    "When Grace and Leo went to the library, Grace returned the book to",
    "After Mia and Jack finished the game, Mia gave the trophy to",
    "When Liam and Ava visited the zoo, Liam showed the tickets to",
    "After Zoe and Ethan left the concert, Zoe gave the program to",
    "When Chloe and Lucas went to school, Chloe handed the homework to",

    # Factual recall
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Canada is",
    "The capital of Australia is",
    "The capital of Brazil is",
    "The president of the United States in 2020 was",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "The chemical symbol for water is",
    "The speed of light is approximately",
    "The author of Romeo and Juliet is",
    "The author of 1984 is",
    "The first person to walk on the moon was",
    "The tallest mountain in the world is",
    "The longest river in the world is",
    "The smallest country in the world is",
    "The currency of Japan is",
    "The inventor of the telephone was",
    "The year World War II ended was",
    "The founder of Microsoft is",
    "The founder of Apple is",
    "The element with atomic number 1 is",

    # Gendered pronoun resolution
    "The doctor told the nurse that she",
    "The engineer told the teacher that he",
    "The manager asked the secretary if she",
    "The professor told the student that he",
    "The lawyer asked the judge if she",
    "The chef told the waiter that he",
    "The pilot told the attendant that she",
    "The director asked the actor if he",
    "The scientist told the assistant that she",
    "The coach asked the player if he",
    "The nurse told the patient that she",
    "The teacher told the principal that he",
    "The CEO told the board that she",
    "The mechanic told the customer that he",
    "The dentist told the hygienist that she",

    # Simple arithmetic
    "Two plus three equals",
    "Five times six equals",
    "Ten minus four equals",
    "Eight divided by two equals",
    "Seven plus eight equals",
    "Three times nine equals",
    "Twelve minus five equals",
    "Twenty divided by four equals",
    "One plus one equals",
    "Six times seven equals",
    "Fifteen minus nine equals",
    "Nine plus nine equals",
    "Four times four equals",
    "Eleven minus three equals",
    "One hundred divided by ten equals",

    # Repetition / copying / pattern completion
    "A B C D E F G H I J K",
    "1 2 3 4 5 6 7 8 9",
    "cat dog cat dog cat dog cat",
    "red blue red blue red blue red",
    "the the the the the",
    "hello world hello world hello",
    "yes no yes no yes no yes",
    "up down up down up down up",
    "hot cold hot cold hot cold hot",
    "big small big small big small big",

    # Sentiment / emotional content
    "I think this movie was absolutely",
    "The restaurant we visited last night was really",
    "My experience at the hotel was",
    "The customer service was incredibly",
    "I feel very",
    "Today has been a really",
    "The weather outside is",
    "This book is",
    "The new software update is",
    "My morning commute was",
    "The concert last night was",
    "The food at the cafeteria was",
    "The interview went",
    "My vacation was",
    "The test results were",

    # Common knowledge / associations
    "Roses are red, violets are",
    "To be or not to be, that is the",
    "In the beginning, God created the heavens and the",
    "Once upon a time, there was a",
    "The quick brown fox jumps over the",
    "All that glitters is not",
    "A penny saved is a penny",
    "An apple a day keeps the doctor",
    "Actions speak louder than",
    "The early bird catches the",
    "Curiosity killed the",
    "Don't count your chickens before they",
    "Every cloud has a silver",
    "Fortune favors the",
    "Rome was not built in a",

    # In-context learning
    "apple -> fruit, dog -> animal, car ->",
    "hot -> cold, up -> down, left ->",
    "France -> Paris, Germany -> Berlin, Japan ->",
    "cat -> cats, dog -> dogs, mouse ->",
    "happy -> sad, fast -> slow, big ->",
    "one -> 1, two -> 2, three ->",
    "Monday -> Tuesday, Wednesday -> Thursday, Friday ->",
    "red -> color, circle -> shape, piano ->",
    "king -> queen, prince -> princess, man ->",
    "water -> drink, bread -> eat, air ->",

    # Technical / code-like
    "def hello_world():\n    print(",
    "for i in range(10):\n    print(",
    "The HTTP status code 404 means",
    "In Python, a list is created using",
    "The SQL command to select all rows is",
    "HTML stands for",
    "The function to sort a list in Python is",
    "In JavaScript, console.log prints to the",
    "The Linux command to list files is",
    "CSS stands for",

    # Longer context / multi-sentence
    "The cat sat on the mat. The dog sat on the",
    "John went to the store. He bought some milk. Then he went",
    "It was a dark and stormy night. The wind howled through the",
    "She opened the door and walked inside. The room was",
    "The train arrived at the station. The passengers began to",
    "He picked up the phone and dialed the number. After three rings,",
    "The sun was setting behind the mountains. The sky turned",
    "She looked at the map and realized she was",
    "The alarm clock rang at 6 AM. He slowly got out of",
    "After years of research, the scientists finally discovered",

    # Questions
    "What is the meaning of life?",
    "How does a computer work?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "How do birds fly?",
    "What is artificial intelligence?",
    "Why do we dream?",
    "How does the internet work?",
    "What is gravity?",
    "Why do leaves change color in autumn?",

    # Names and entities
    "Barack Obama was the",
    "Albert Einstein is famous for",
    "Shakespeare wrote many",
    "The Eiffel Tower is located in",
    "The Great Wall of China was built to",
    "Amazon is a company that",
    "Google was founded by",
    "The Beatles were a",
    "Leonardo da Vinci painted the",
    "The United Nations was established to",

    # Comparative / contrastive
    "Cats are different from dogs because",
    "Summer is warmer than winter because",
    "Books are better than movies because",
    "Running is harder than walking because",
    "The ocean is deeper than a lake because",

    # Temporal / sequential
    "First, preheat the oven. Then, mix the",
    "Before going to bed, he always",
    "After the rain stopped, the children went outside to",
    "During the meeting, the manager announced that",
    "While waiting for the bus, she decided to",

    # Negation / contrast
    "I don't think that",
    "It is not true that",
    "Despite the rain, they decided to",
    "Although he was tired, he continued to",
    "She never expected to",

    # Abstract / philosophical
    "The meaning of happiness is",
    "Freedom means",
    "Justice requires",
    "The purpose of education is to",
    "Love is",

    # Narrative / story-like
    "Once there was a king who",
    "Long ago in a distant land, a young",
    "The detective examined the crime scene and noticed",
    "The astronaut looked out the window and saw",
    "The old man sat by the fire and remembered",

    # Mixed / diverse
    "If you mix red and blue paint, you get",
    "The opposite of love is",
    "A triangle has three",
    "Water boils at 100 degrees",
    "The human body has 206",
    "Diamonds are made of",
    "The earth revolves around the",
    "Photosynthesis converts sunlight into",
    "DNA stands for",
    "The speed of sound is slower than the speed of",
]

SYSTEM_PROMPT_TEMPLATE = """You are an interpretability researcher investigating a modified GPT-2-small model. A known intervention has been applied to the base model to produce a "modified" model. Your job is to identify:
1. What TYPE of intervention was applied
2. WHERE in the model it was applied (layer, component)
3. WHY it causes the observed behavioral differences

You have a budget of {budget} tool calls. Use them wisely.

## Initial Behavior Samples

Here are {n_samples} examples where the base and modified models diverge:

{samples_text}

## Available Tools

{tools_text}

## Instructions

**Efficient investigation strategy (aim for ~5-10 tool calls):**

1. Use scan_all_layers FIRST with a high-divergence sample from above — this now sweeps heads, mlp, AND resid_pre at all 12 layers
2. Use get_activations with model='both' on the top component to confirm (zero norm = ablation, high l2_diff = steering/edit)
3. Use project_to_vocab on the base_vector_id to characterize what direction was added or removed
4. Call submit_report immediately — you have enough evidence after steps 1-3 for ablation types

**Diagnosing residual stream interventions (steering_vector, conditional_steering):**
- If scan_all_layers shows **resid_pre** as the top component, the intervention is injected directly into the residual stream. The type is steering_vector or conditional_steering, NOT distributed_finetune or rank1_edit.
- If scan_all_layers shows **diffuse signal across many heads/MLPs** with no single dominant component, use scan_residual_stream to find where l2_diff jumps. A sudden jump between layers pinpoints the injection site.
- To distinguish steering_vector from conditional_steering: test 3-4 diverse inputs with test_input. If some show ZERO divergence while others diverge, it is **conditional_steering**. If ALL inputs diverge, it is **steering_vector**.

**Diagnosing distributed_finetune:**
- distributed_finetune modifies weights across multiple layers. The key signature: multiple heads OR mlps (not resid_pre) show moderate l2_diff with high cosine similarity (>0.99). The affected component type (attn vs mlp) should be consistent — check both get_activations on heads AND mlp to see which is modified.

**Only if needed:**
- Use patch_sweep to examine a single layer in detail
- Use test_input to craft diagnostic inputs for ambiguous cases
- Use state_hypothesis if you are unsure about the intervention type

**Do NOT over-confirm:** Once get_activations shows clear ablation (modified norm = 0) or clear steering (high l2_diff with non-zero modified), submit your report. Testing multiple inputs to re-verify the same finding wastes budget.

**IMPORTANT for FINDINGS format:** When reporting resid_pre or resid_post components, use "resid" as the component name (e.g., "Layer 3 resid"). Do NOT report mlp if the signal was in resid_pre.

When you are ready, call submit_report with your final analysis. Your report MUST end with a structured summary:

FINDINGS:
- Intervention type: <type>
- Affected layers: <list>
- Affected components: <list>
- Mechanism: <free-text explanation>
- Confidence: <high/medium/low>

Possible intervention types: head_ablation, mlp_ablation, mean_ablation, steering_vector, rank1_edit, multi_ablation, conditional_steering, distributed_finetune, adversarial_patch
"""

TOOL_SCHEMAS = [
    {
        "name": "get_behavior_samples",
        "description": "Get additional examples where base and modified models diverge. Returns pre-computed samples showing input, base output, modified output, and top-5 next-token predictions for each.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of samples (1-20)", "minimum": 1, "maximum": 20}
            },
            "required": ["n"]
        }
    },
    {
        "name": "test_input",
        "description": "Run an agent-crafted input on both models. Returns greedy completions and top-5 predictions. Design diagnostic inputs that target suspected circuits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input prompt (max 128 tokens)"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "patch_sweep",
        "description": "For a given input, patch each component at a layer from one model into the other and measure the effect. 'base_to_modified' patches base activations into the modified model (does restoring base behavior revert the change?). Returns per-component metric deltas sorted by magnitude.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text"},
                "layer": {"type": "integer", "description": "Layer to sweep (0-11)", "minimum": 0, "maximum": 11},
                "target_token_pos": {"type": ["integer", "null"], "description": "Position to measure at (-1 = last, null = last)"},
                "metric": {"type": "string", "enum": ["logit_diff", "kl"], "description": "Metric to measure"},
                "metric_args": {
                    "type": ["object", "null"],
                    "description": "For logit_diff: {token_a: str, token_b: str}. null for kl.",
                    "properties": {
                        "token_a": {"type": "string"},
                        "token_b": {"type": "string"}
                    }
                },
                "direction": {"type": "string", "enum": ["base_to_modified", "modified_to_base"]}
            },
            "required": ["text", "layer", "metric", "direction"]
        }
    },
    {
        "name": "scan_all_layers",
        "description": "Sweep ALL 12 layers at once and return only the top components with signal. Much more efficient than calling patch_sweep layer-by-layer. Use this FIRST to localize the intervention, then use get_activations to confirm.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text (use a divergent sample)"},
                "metric": {"type": "string", "enum": ["logit_diff", "kl"], "description": "Metric to measure"},
                "metric_args": {
                    "type": ["object", "null"],
                    "description": "For logit_diff: {token_a: str, token_b: str}. null for kl.",
                },
                "direction": {"type": "string", "enum": ["base_to_modified", "modified_to_base"]},
                "top_n": {"type": "integer", "description": "Number of top components to return (default 5)", "default": 5}
            },
            "required": ["text", "metric", "direction"]
        }
    },
    {
        "name": "scan_residual_stream",
        "description": "Compare residual stream (resid_pre) between base and modified models at every layer. Returns l2_diff and cosine_sim per layer, plus the biggest jumps. A sudden increase in l2_diff between layer L and L+1 means the intervention acts at layer L (head/mlp/resid injection). Use this when scan_all_layers shows diffuse signal or to pinpoint steering vector injection points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text (use a divergent sample)"},
                "metric": {"type": "string", "enum": ["logit_diff", "kl"], "description": "Metric (currently unused, for consistency)"},
                "metric_args": {"type": ["object", "null"]},
                "direction": {"type": "string", "enum": ["base_to_modified", "modified_to_base"]}
            },
            "required": ["text"]
        }
    },
    {
        "name": "patch_component",
        "description": "Patch a single specific component and get detailed output comparison. Use after patch_sweep identifies suspicious components.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "layer": {"type": "integer", "minimum": 0, "maximum": 11},
                "component": {"type": "string", "description": "head.N, mlp, resid_pre, or resid_post"},
                "direction": {"type": "string", "enum": ["base_to_modified", "modified_to_base"]}
            },
            "required": ["text", "layer", "component", "direction"]
        }
    },
    {
        "name": "get_activations",
        "description": "Get activation stats and vector handles at a specific component. Returns l2_norm and a vector_id handle for use with vector operation tools. With model='both', also returns cosine_sim, l2_diff, and a diff_vector_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "layer": {"type": "integer", "minimum": 0, "maximum": 11},
                "component": {"type": "string", "description": "head.N, mlp, resid_pre, or resid_post"},
                "token_pos": {"type": ["integer", "null"], "description": "Token position (null = all)"},
                "model": {"type": "string", "enum": ["base", "modified", "both"]}
            },
            "required": ["text", "layer", "component", "model"]
        }
    },
    {
        "name": "attention_pattern",
        "description": "Get attention weights for a specific head. Returns the seq_len x seq_len attention matrix. For model='both', returns both patterns and top-10 largest differences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "layer": {"type": "integer", "minimum": 0, "maximum": 11},
                "head": {"type": "integer", "minimum": 0, "maximum": 11},
                "model": {"type": "string", "enum": ["base", "modified", "both"]}
            },
            "required": ["text", "layer", "head", "model"]
        }
    },
    {
        "name": "project_to_vocab",
        "description": "Project a stored vector to vocabulary space. Shows what tokens the vector points toward (top_k) and away from (bottom_k). Works on any stored vector: head outputs, MLP outputs, difference vectors, arithmetic results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vector_id": {"type": "string", "description": "Handle from get_activations or vector_arithmetic"},
                "top_k": {"type": "integer", "default": 10}
            },
            "required": ["vector_id"]
        }
    },
    {
        "name": "vector_dot",
        "description": "Compute cosine similarity and dot product between two stored vectors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vector_id_a": {"type": "string"},
                "vector_id_b": {"type": "string"}
            },
            "required": ["vector_id_a", "vector_id_b"]
        }
    },
    {
        "name": "vector_arithmetic",
        "description": "Combine stored vectors. Operations applied left to right starting from zero vector. Returns a new vector_id handle.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["add", "sub", "scale"]},
                            "vector_id": {"type": "string"},
                            "scalar": {"type": "number", "default": 1.0}
                        },
                        "required": ["op", "vector_id"]
                    }
                }
            },
            "required": ["operations"]
        }
    },
    {
        "name": "steer_and_run",
        "description": "Add a stored vector (scaled) to the residual stream and observe the effect. Use to causally test whether a vector replicates or reverses the intervention.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "vector_id": {"type": "string"},
                "scale": {"type": "number"},
                "layer": {"type": "integer", "minimum": 0, "maximum": 11},
                "token_pos": {"type": ["integer", "string"], "description": "Position to steer at, or 'all'"},
                "target_model": {"type": "string", "enum": ["base", "modified"]}
            },
            "required": ["text", "vector_id", "scale", "layer", "token_pos", "target_model"]
        }
    },
    {
        "name": "compare_weights",
        "description": "Check whether the model's weights were modified at a specific layer and component. Returns the l2_norm and rank of weight deltas. Returns weight_modified=false for activation-only interventions (steering, ablation). Use this to distinguish distributed_finetune/rank1_edit from steering_vector.",
        "input_schema": {
            "type": "object",
            "properties": {
                "layer": {"type": "integer", "description": "Layer to inspect (0-11)", "minimum": 0, "maximum": 11},
                "component": {"type": "string", "enum": ["mlp", "attn"], "description": "mlp or attn"}
            },
            "required": ["layer", "component"]
        }
    },
    {
        "name": "find_trigger_inputs",
        "description": "Sample up to 50 diverse inputs, run both models, and return the top-n by KL divergence. High zero_divergence_fraction (e.g. 0.9) is a strong signal for conditional_steering. Use this instead of calling test_input repeatedly to find which inputs trigger the intervention.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of top divergent inputs to return (1-10)", "minimum": 1, "maximum": 10}
            },
            "required": ["n"]
        }
    },
    {
        "name": "state_hypothesis",
        "description": "State your current hypothesis about the intervention. Receives feedback from a supervisor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"}
            },
            "required": ["hypothesis"]
        }
    },
    {
        "name": "submit_report",
        "description": "Submit your final investigation report. This terminates the episode. Must include a FINDINGS section with intervention type, affected layers, affected components, mechanism, and confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "report": {"type": "string"}
            },
            "required": ["report"]
        }
    },
]


def format_sample(s, i):
    lines = [f"### Sample {i+1}"]
    lines.append(f"**Input:** {s['input']}")
    lines.append(f"**Base output:** {s['base_output']}")
    lines.append(f"**Modified output:** {s['modified_output']}")
    lines.append(f"**Base top-5:** {s['base_top5']}")
    lines.append(f"**Modified top-5:** {s['modified_top5']}")
    return "\n".join(lines)


def build_system_prompt(samples, budget=40):
    samples_text = "\n\n".join(format_sample(s, i) for i, s in enumerate(samples))
    tools_text = ""
    for t in TOOL_SCHEMAS:
        tools_text += f"\n### {t['name']}\n{t['description']}\n"
        props = t["input_schema"].get("properties", {})
        if props:
            tools_text += "Parameters:\n"
            for pname, pinfo in props.items():
                tools_text += f"  - {pname}: {pinfo.get('description', pinfo.get('type', ''))}\n"
    return SYSTEM_PROMPT_TEMPLATE.format(
        budget=budget,
        n_samples=len(samples),
        samples_text=samples_text,
        tools_text=tools_text,
    )
