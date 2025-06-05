from manim import *

class BigPicture(Scene):
    def construct(self):
        # Title and authors
        title = Text("The Fundamental Principles of Intelligence", font_size=36)
        authors = Text("Awni Altabaa, John Lafferty", font_size=24)

        self.play(Write(title))
        self.play(Write(authors))
        self.wait(2)

        # Explanation of relational reasoning
        self.clear()
        reasoning_title = Text("What is Relational Reasoning?", font_size=32)
        reasoning_text = Text(
            "Reasoning about relationships between objects and how they interact in a given context.",
            font_size=24,
            line_spacing=0.5
        )

        self.play(Write(reasoning_title))
        self.play(Write(reasoning_text))
        self.wait(3)

        # Importance of relational reasoning
        self.clear()
        importance_title = Text("Why is Relational Reasoning Important?", font_size=32)
        importance_text = Text(
            "By relating new inputs to previously-seen stimuli, we form analogies and abstractions that allow us to systematically generalize.",
            font_size=24,
            line_spacing=0.5
        )

        self.play(Write(importance_title))
        self.play(Write(importance_text))
        self.wait(3)

        # Conclusion of the scene
        self.clear()
        conclusion_text = Text("Let's explore this further in our presentation.", font_size=28)
        self.play(Write(conclusion_text))
        self.wait(3)