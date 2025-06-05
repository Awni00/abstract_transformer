from manim import *

class Conclusion(Scene):
    def construct(self):
        # Title
        title = Text("Concluding Remarks", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Main content
        remarks = [
            "Relational reasoning is a core facet of human intelligence,",
            "underpinning abilities for analogy, abstraction, and generalization.",
            "It is likely an important component of artificial intelligence as well.",
            "In this work, we took a step towards developing neural architectures",
            "with enhanced relational processing capabilities, while retaining",
            "powerful sensory processing."
        ]

        remarks_text = VGroup(*[Text(remark, font_size=36) for remark in remarks]).arrange(DOWN, buff=0.5)
        self.play(Write(remarks_text))
        self.wait(2)

        # Future Work
        future_work_title = Text("Future Work", font_size=48)
        self.play(Transform(title, future_work_title))
        self.wait(1)

        future_remarks = [
            "Interpretability:",
            "• How is DAT learning to use its relational processing mechanisms?",
            "• Can specific circuits be identified?",
            "• How does DAT achieve improved data efficiency in different tasks?",
            "Iterate & tweak architecture; find good choices for hyperparameters.",
            "Computational considerations: optimize implementation."
        ]

        future_remarks_text = VGroup(*[Text(remark, font_size=36) for remark in future_remarks]).arrange(DOWN, buff=0.5)
        self.play(Write(future_remarks_text))
        self.wait(2)

        # Thank You
        thank_you = Text("Thank You", font_size=48)
        self.play(Transform(title, thank_you))
        self.wait(2)

        # Discussion prompt
        discussion_prompt = Text("Discussion Time...", font_size=36)
        self.play(Write(discussion_prompt))
        self.wait(2)

        self.clear()