"""
Data ingestion script for preparing evaluation datasets.

Generates structured test prompts for predictable (Alpaca) and open-ended (WritingPrompts) domains.
"""

import json
from pathlib import Path
from typing import List, Dict


def prepare_datasets(output_path: str = "data/test_prompts.json") -> List[Dict[str, str]]:
    """
    Constructs and persists the test dataset with predictable and open-ended prompts.

    Args:
        output_path: Path where data/test_prompts.json will be saved.

    Returns:
        List of prompt dictionaries with id, domain, and prompt.
    """
    predictable_prompts = [
        "Translate to French: The meeting is scheduled for 3pm.",
        "Translate to Spanish: Good morning, how are you today?",
        "Translate to German: Thank you very much for your help.",
        "Convert the following temperature from Celsius to Fahrenheit: 25 degrees Celsius.",
        "List the primary colors in standard color theory.",
        "What is the capital city of France?",
        "What is the chemical formula for water?",
        "Calculate the product of 12 and 15.",
        "Complete the sequence: 2, 4, 6, 8, 10,",
        "Extract the date from this sentence: The event will take place on October 14, 2026.",
        "Classify the sentiment of this review as Positive or Negative: The food was absolutely delicious and the service was superb.",
        "Classify the sentiment of this review as Positive or Negative: The flight was delayed by five hours and my luggage was lost.",
        "Format the following telephone number with hyphens: 1234567890",
        "Name the four cardinal directions on a compass.",
        "What is the square root of 64?",
        "Identify the language of the following phrase: 'Bonjour tout le monde'",
        "Alphabetize the following list of words: apple, zebra, banana, orange, grape.",
        "Translate to Italian: Where is the nearest train station?",
        "State the boiling point of water in degrees Celsius at standard atmospheric pressure.",
        "Which planet is known as the Red Planet in our solar system?"
    ]

    open_ended_prompts = [
        "Write a surprising twist ending for a short story about a lighthouse keeper.",
        "Describe a bustling steampunk marketplace at twilight from the perspective of an owl.",
        "Compose an enigmatic riddle about time and lost memories.",
        "Write the opening scene of a sci-fi novel where silence is used as currency.",
        "Imagine a conversation between an ancient tree and the first satellite launched into orbit.",
        "Describe the sensory experience of walking through a library where books whisper their secrets.",
        "Draft a poetic monologue of an ocean wave reaching the shore during a stormy night.",
        "Create a dialogue between two rival chefs who secretly respect each other's culinary genius.",
        "Write a vivid description of an alien city carved entirely out of bioluminescent crystal.",
        "Invent a philosophical fable about a clockmaker who discovered how to pause a single second.",
        "Describe an unexpected friendship between an eccentric inventor and a runaway artificial intelligence.",
        "Write a scene where the stars in the night sky suddenly rearrange themselves into ancient symbols.",
        "Depict the internal thoughts of a deep-sea diver discovering a submerged metropolis.",
        "Compose a surreal dream sequence where gravity behaves like water.",
        "Write a tale about a forgotten melody that has the power to restore forgotten memories.",
        "Describe the dawn breaking over a post-apocalyptic garden tended by gentle automata.",
        "Write a conversation between the sun and the moon during a total solar eclipse.",
        "Describe an antique mirror that reflects not the person standing in front of it, but their future choices.",
        "Create an intriguing backstory for a wandering cartographer who maps dreams rather than continents.",
        "Write the final diary entry of an explorer who reached the edge of an infinite labyrinth."
    ]

    dataset: List[Dict[str, str]] = []

    # Include benchmark prompts p1 and p2 as top items
    dataset.append({
        "id": "p1",
        "domain": "alpaca",
        "prompt": "Translate to French: The meeting is scheduled for 3pm."
    })
    dataset.append({
        "id": "p2",
        "domain": "writing_prompts",
        "prompt": "Write a surprising twist ending for a short story about a lighthouse keeper."
    })

    idx = 3
    # Add predictable prompts
    for p in predictable_prompts[1:]:
        dataset.append({
            "id": f"p{idx}",
            "domain": "alpaca",
            "prompt": p
        })
        idx += 1

    # Add open-ended prompts
    for p in open_ended_prompts[1:]:
        dataset.append({
            "id": f"p{idx}",
            "domain": "writing_prompts",
            "prompt": p
        })
        idx += 1

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(dataset)} prompts across 'alpaca' and 'writing_prompts' domains to {output_path}")
    return dataset


if __name__ == "__main__":
    prepare_datasets()
