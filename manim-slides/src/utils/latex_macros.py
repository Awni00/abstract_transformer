from manim import *

def create_math_tex(tex_string, color=WHITE, font_size=24):
    """Helper function to create consistent MathTex objects"""
    return MathTex(tex_string, color=color, font_size=font_size)

# Mathematical notation functions
def rel_attention():
    return r"\mathrm{RelationalAttention}"

def attention():
    return r"\mathrm{Attention}"

def self_attention():
    return r"\mathrm{SelfAttention}"

def mlp():
    return r"\mathrm{MLP}"

def layer_norm():
    return r"\mathrm{LayerNorm}"

def softmax():
    return r"\mathrm{Softmax}"

def dropout():
    return r"\mathrm{Dropout}"

def relu():
    return r"\mathrm{ReLU}"

def gelu():
    return r"\mathrm{GELU}"

def concat():
    return r"\mathrm{concat}"

# Dimension variables
def d_model():
    return r"d_{\mathrm{model}}"

def n_layers():
    return r"n_{\mathrm{layers}}"

def d_ff():
    return r"d_{\mathrm{ff}}"

def d_key():
    return r"d_{\mathrm{key}}"

# Vector and matrix notation
def bold_r():
    return r"\bm{r}"

def bold_x():
    return r"\bm{x}"

def bold_y():
    return r"\bm{y}"

def bold_s():
    return r"\bm{s}"

# Create a dictionary mapping LaTeX commands to their equivalents
LATEX_MACROS = {
    r"\RelAttn": rel_attention(),
    r"\Attn": attention(),
    r"\SelfAttn": self_attention(),
    r"\MLP": mlp(),
    r"\LayerNorm": layer_norm(),
    r"\Softmax": softmax(),
    r"\Dropout": dropout(),
    r"\ReLU": relu(),
    r"\GELU": gelu(),
    r"\concat": concat(),
    r"\dmodel": d_model(),
    r"\nlayers": n_layers(),
    r"\dff": d_ff(),
    r"\dkey": d_key(),
}

def process_latex_string(latex_str):
    """Process a LaTeX string by replacing custom macros"""
    for macro, replacement in LATEX_MACROS.items():
        latex_str = latex_str.replace(macro, replacement)
    return latex_str

def create_equation(latex_str, **kwargs):
    """Create a MathTex object with processed macros"""
    processed_str = process_latex_string(latex_str)
    return MathTex(processed_str, **kwargs)