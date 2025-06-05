from manim import *

class Introduction(Scene):
    def construct(self):
        title = Text("Disentangling and Integrating Relational and Sensory Information in Transformer Architectures", font_size=36)
        authors = Text("Awni Altabaa, John Lafferty", font_size=24)

        self.play(Write(title))
        self.wait(2)
        self.play(Write(authors))
        self.wait(2)

        self.play(FadeOut(title), FadeOut(authors))