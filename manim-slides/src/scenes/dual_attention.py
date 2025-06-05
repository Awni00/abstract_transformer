from manim import *

class DualAttentionScene(Scene):
    def construct(self):
        # Title and Authors
        title = Text("Dual Attention Transformer Architecture", font_size=48)
        authors = Text("Awni Altabaa, John Lafferty", font_size=36)
        self.play(Write(title))
        self.play(Write(authors.next_to(title, DOWN)))
        self.wait(2)
        self.clear()

        # Introduction to Dual Attention
        intro_text = Text("Introducing Dual Attention", font_size=36)
        self.play(Write(intro_text))
        self.wait(2)
        self.clear()

        # Explanation of Dual Attention
        explanation = Tex(
            r"""
            \textbf{Dual Attention} is a variant of multi-head attention with two types of attention heads:
            \begin{itemize}
                \item \textit{Sensory Attention}
                \item \textit{Relational Attention}
            \end{itemize}
            """,
            font_size=36
        )
        self.play(Write(explanation))
        self.wait(4)
        self.clear()

        # Visual Representation of Dual Attention
        dual_attention_diagram = ImageMobject("path/to/your/dual_attention_diagram.png").scale(0.5)
        self.play(FadeIn(dual_attention_diagram))
        self.wait(3)
        self.clear()

        # Summary of the Architecture
        summary = Tex(
            r"""
            The Dual Attention Transformer architecture introduces explicit relational processing mechanisms,
            while retaining sensory processing capabilities.
            """,
            font_size=36
        )
        self.play(Write(summary))
        self.wait(4)
        self.clear()

        # Conclusion Slide
        conclusion = Text("Thank You!", font_size=48)
        self.play(Write(conclusion))
        self.wait(2)