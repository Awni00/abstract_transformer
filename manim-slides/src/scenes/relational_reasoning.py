from manim import *

class RelationalReasoning(Scene):
    def construct(self):
        # Title Slide
        title = Text("What is Relational Reasoning?", font_size=48)
        self.play(Write(title))
        self.wait(2)
        self.clear()

        # Definition of Relational Reasoning
        definition = Text(
            "Reasoning about relationships between objects and how they interact in a given context.",
            font_size=36,
            line_spacing=1.5
        )
        self.play(Write(definition))
        self.wait(3)
        self.clear()

        # Importance of Relational Reasoning
        importance = Text(
            "Clue to its importance: Humans have a natural ability (and a preference) to do relational reasoning.",
            font_size=36,
            line_spacing=1.5
        )
        self.play(Write(importance))
        self.wait(3)
        self.clear()

        # Example: SET! Card Game
        set_game_image = ImageMobject("assets/images/set_game.png").scale(0.5)
        self.play(FadeIn(set_game_image))
        self.wait(2)
        self.clear()

        # Example: Relational Games
        relational_games = Text(
            "Example: Relational Games (Shanahan et al. 2020)",
            font_size=36,
            line_spacing=1.5
        )
        self.play(Write(relational_games))
        self.wait(3)
        self.clear()

        # Returning to the original question
        question = Text(
            "Why should we care about relational reasoning?",
            font_size=36,
            line_spacing=1.5
        )
        self.play(Write(question))
        self.wait(3)
        self.clear()

        # Conclusion Slide
        conclusion = Text(
            "By relating new inputs to previously-seen stimuli, we form analogies and abstractions that allow us to systematically generalize.",
            font_size=36,
            line_spacing=1.5
        )
        self.play(Write(conclusion))
        self.wait(3)
        self.clear()