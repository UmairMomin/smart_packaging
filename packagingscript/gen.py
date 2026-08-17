import tkinter as tk
from tkinter import scrolledtext

# --- PROMPT TEMPLATES ---
coin_prompt = """
 low-resolution photorealistic image taken from a perfectly top-down, bird’s-eye view.
The camera is exactly perpendicular to the surface (90 degrees), with no perspective distortion.

A single {product} is placed flat on a neutral, textured surface.
The entire product is fully visible within the frame.  place exactly one Indian 10 Rupees coin (Indian currency).

Introduce realistic smartphone capture noise:
- Slight camera sensor grain
- Mild uneven indoor lighting
- Soft natural shadows near the edges
- Semi-sharp focus (not studio sharp)
The coin is placed very close to the product, touching or within 5 mm.
Both objects are fully visible.
No tilt, no occlusion, no reflections, no extra objects.
No hands, no people.
The image should look like a real user-captured photo suitable for ML training.
The overall image aspect ratio must vary naturally between generations.
 Each generation should randomly use a different realistic smartphone aspect ratio such as:
1:1 (square)
4:3
3:4
16:9
9:16
5:4
2:3
The product and reference object  must remain fully visible regardless of aspect ratio.
Framing should adjust naturally to the selected aspect ratio while preserving:
Perfect top-down orientation (90 degrees)
No perspective distortion
No cropping of the product or square
Realistic spacing and composition
The background remains the same as mentioned, but framing and composition should vary slightly between generations to simulate real user capture behavior.
Do not repeat the same aspect ratio across generations unless randomly selected.

"""

card_prompt = """
A low-resolution photorealistic image taken from a perfectly top-down, bird’s-eye view.
The camera is exactly perpendicular to the surface (90 degrees), with no perspective distortion.

A single {product} is placed flat on a neutral, textured surface.
The entire product is fully visible within the frame. As the ONLY reference object, place exactly one ATM or credit card.

The card must be rectangular, standard credit-card size, with rounded corners use indian debit card.

Introduce realistic smartphone capture noise:
- Slight camera sensor grain
- Mild uneven indoor lighting
- Soft natural shadows near the edges
- Semi-sharp focus (not studio sharp)
The Atm card is placed very close to the product, touching or within 5 mm.
Both objects are fully visible.
No tilt, no occlusion, no reflections, no extra objects.
No hands, no people.
The image should look like a real user-captured photo suitable for ML training.The overall image aspect ratio must vary naturally between generations.
 Each generation should randomly use a different realistic smartphone aspect ratio such as:
1:1 (square)
4:3
3:4
16:9
9:16
5:4
2:3
The product and reference object must remain fully visible regardless of aspect ratio.
Framing should adjust naturally to the selected aspect ratio while preserving:
Perfect top-down orientation (90 degrees)
No perspective distortion
No cropping of the product or square
Realistic spacing and composition
The background remains the but framing and composition should vary slightly between generations to simulate real user capture behavior.
Do not repeat the same aspect ratio across generations unless randomly selected.

"""

square_prompt = """
A low-resolution photorealistic image captured from a perfectly top-down, bird’s-eye view.
 The camera is exactly perpendicular to the surface (90 degrees), with zero perspective distortion.
The entire background surface is a single large white sheet of paper covering the full frame.
 No other background texture is visible.
A single {product} is placed flat on this white paper surface.
 The entire product is fully visible within the frame.
On the same white paper, draw exactly one small black square using a ruler and black pen.
 The square must be clean, precise, and geometrically accurate.
 All four sides must be equal length.
 Lines must be straight, uniform thickness, and appear measured — not hand-sketched.
The square must appear very small relative to the product, clearly reading as a physical measurement reference.
 It should visually look fingertip-sized and significantly smaller than the product.
The square must be placed immediately next to the product, touching or within 5 mm distance.
Both the product and the entire square must be fully visible.
Introduce realistic smartphone capture qualities:
Slight sensor grain
Mild uneven indoor lighting
Soft natural shadows around the product edges
Semi-sharp focus (not studio-level sharpness)
No tilt, no angle, no perspective distortion.
 No reflections, no extra objects, no text, no markings other than the single square.
 No hands, no people.
The image must look like a real user-captured smartphone photo suitable for machine learning training.
The overall image aspect ratio must vary naturally between generations.
 Each generation should randomly use a different realistic smartphone aspect ratio such as:
1:1 (square)
4:3
3:4
16:9
9:16
5:4
2:3
The product and reference object must remain fully visible regardless of aspect ratio.
Framing should adjust naturally to the selected aspect ratio while preserving:
Perfect top-down orientation (90 degrees)
No perspective distortion
No cropping of the product or square
Realistic spacing and composition
The background remains the same white paper surface, but framing and composition should vary slightly between generations to simulate real user capture behavior.
Do not repeat the same aspect ratio across generations unless randomly selected.

"""


# --- GUI FUNCTION ---
def copy_to_clipboard(textbox):
    root.clipboard_clear()
    root.clipboard_append(textbox.get("1.0", tk.END))


def generate_prompts():
    product = product_entry.get().strip()
    if not product:
        return

    coin_box.delete(1.0, tk.END)
    card_box.delete(1.0, tk.END)
    grid_box.delete(1.0, tk.END)

    coin_box.insert(tk.END, coin_prompt.format(product=product))
    card_box.insert(tk.END, card_prompt.format(product=product))
    grid_box.insert(tk.END, square_prompt.format(product=product))


# --- GUI SETUP ---
root = tk.Tk()
root.title("Dataset Prompt Generator")
root.geometry("750x700")

title = tk.Label(root, text="Product Prompt Generator", font=("Arial", 16))
title.pack(pady=10)

# Product input
input_frame = tk.Frame(root)
input_frame.pack(pady=5)

tk.Label(input_frame, text="Product Name: ").pack(side=tk.LEFT)
product_entry = tk.Entry(input_frame, width=30)
product_entry.pack(side=tk.LEFT, padx=5)

tk.Button(root, text="Generate Prompts", command=generate_prompts).pack(pady=10)

# --- COIN PROMPT ---
coin_frame = tk.Frame(root)
coin_frame.pack(fill="both", padx=10, pady=5)

tk.Label(coin_frame, text="Coin Prompt", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
tk.Button(coin_frame, text="Copy", command=lambda: copy_to_clipboard(coin_box)).pack(side=tk.RIGHT)

coin_box = scrolledtext.ScrolledText(root, height=8, wrap=tk.WORD)
coin_box.pack(fill="both", padx=10)

# --- CARD PROMPT ---
card_frame = tk.Frame(root)
card_frame.pack(fill="both", padx=10, pady=5)

tk.Label(card_frame, text="Card Prompt", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
tk.Button(card_frame, text="Copy", command=lambda: copy_to_clipboard(card_box)).pack(side=tk.RIGHT)

card_box = scrolledtext.ScrolledText(root, height=8, wrap=tk.WORD)
card_box.pack(fill="both", padx=10)

# --- GRID PROMPT ---
grid_frame = tk.Frame(root)
grid_frame.pack(fill="both", padx=10, pady=5)

tk.Label(grid_frame, text="2x2 Grid Prompt", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
tk.Button(grid_frame, text="Copy", command=lambda: copy_to_clipboard(grid_box)).pack(side=tk.RIGHT)

grid_box = scrolledtext.ScrolledText(root, height=8, wrap=tk.WORD)
grid_box.pack(fill="both", padx=10)

root.mainloop()