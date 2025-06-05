from manim import *

class TransformersScene(Scene):
    def construct(self):
        # Title
        title = Text("How to Imbue Transformers with Explicit Relational Inductive Biases", font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.clear()

        # Content
        content = [
            "1. Attention captures sensory information.",
            "2. Relational attention captures relationships between objects.",
            "3. Two types of attention are necessary for effective processing."
        ]

        # Create a list of bullet points
        bullet_points = VGroup(*[Text(line, font_size=24) for line in content]).arrange(DOWN, buff=0.5)
        self.play(Write(bullet_points))
        self.wait(3)
        self.clear()

        # Explanation of attention types
        attention_types = Text("Types of Attention", font_size=36)
        self.play(Write(attention_types))
        self.wait(2)
        self.clear()

        # Sensory Attention
        sensory_attention = Text("Sensory Attention", font_size=28)
        sensory_attention.next_to(attention_types, DOWN, buff=1)
        self.play(Write(sensory_attention))
        self.wait(2)

        # Relational Attention
        relational_attention = Text("Relational Attention", font_size=28)
        relational_attention.next_to(sensory_attention, DOWN, buff=0.5)
        self.play(Write(relational_attention))
        self.wait(2)

        # Conclusion
        conclusion = Text("Both types of attention are crucial for Transformer architectures.", font_size=24)
        conclusion.next_to(relational_attention, DOWN, buff=1)
        self.play(Write(conclusion))
        self.wait(3)

        # End scene
        self.clear()