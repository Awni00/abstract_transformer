from manim import *

class Experiments(Scene):
    def construct(self):
        # Title Slide
        title = Text("Empirical Investigation", font_size=48)
        self.play(Write(title))
        self.wait(2)
        self.clear()

        # Prelude Slide
        prelude = Text("Prelude: What questions are we trying to answer?", font_size=36)
        self.play(Write(prelude))
        self.wait(2)
        self.clear()

        questions = [
            "How does the DAT perform on synthetic relational benchmarks?",
            "Applicability to complex real-world tasks;",
            "Versatility across data modalities (language & vision)"
        ]
        
        questions_text = VGroup(*[Text(q, font_size=24) for q in questions]).arrange(DOWN, buff=0.5)
        self.play(Write(questions_text))
        self.wait(3)
        self.clear()

        # Standout Slide for Synthetic Relational Benchmarks
        standout = Text("Synthetic Relational Benchmarks: Relational Games (Shanahan et al. 2020)", font_size=36)
        self.play(Write(standout))
        self.wait(2)
        self.clear()

        # Results Slide
        results_title = Text("Synthetic Relational Tasks: Results", font_size=36)
        self.play(Write(results_title))
        self.wait(2)

        # Assuming the results image is in the assets/images directory
        results_image = ImageMobject("assets/images/relgames_learning_curves.png")
        results_image.scale(0.5)
        self.play(FadeIn(results_image))
        self.wait(3)
        self.clear()

        # Standout Slide for Mathematical Problem-Solving
        math_standout = Text("Mathematical Problem-Solving (Seq2Seq)", font_size=36)
        self.play(Write(math_standout))
        self.wait(2)
        self.clear()

        # Task Slide for Mathematical Problem-Solving
        task_title = Text("Mathematical Problem-Solving (Seq2Seq): Task", font_size=36)
        self.play(Write(task_title))
        self.wait(2)

        task_description = Text("Dataset due to Saxton et al. (2019)", font_size=24)
        self.play(Write(task_description))
        self.wait(2)
        self.clear()

        # Results Slide for Mathematical Problem-Solving
        math_results_title = Text("Mathematical Problem-Solving (Seq2Seq): Results", font_size=36)
        self.play(Write(math_results_title))
        self.wait(2)

        # Assuming the results image is in the assets/images directory
        math_results_image = ImageMobject("assets/images/math_accuracy_scaling.png")
        math_results_image.scale(0.5)
        self.play(FadeIn(math_results_image))
        self.wait(3)
        self.clear()

        # Standout Slide for Visual Processing
        visual_standout = Text("Visual Processing (CIFAR)", font_size=36)
        self.play(Write(visual_standout))
        self.wait(2)
        self.clear()

        # Task Slide for Visual Processing
        visual_task_title = Text("Visual Processing (CIFAR): Task", font_size=36)
        self.play(Write(visual_task_title))
        self.wait(2)

        # Assuming the CIFAR image is in the assets/images directory
        cifar_image = ImageMobject("assets/images/cifar10_demo.png")
        cifar_image.scale(0.5)
        self.play(FadeIn(cifar_image))
        self.wait(3)
        self.clear()

        # Results Slide for Visual Processing
        visual_results_title = Text("Visual Processing (CIFAR): Results", font_size=36)
        self.play(Write(visual_results_title))
        self.wait(2)

        visual_results_description = Text("ViT-style encoder-only architecture processing image as sequence of patches", font_size=24)
        self.play(Write(visual_results_description))
        self.wait(3)
        self.clear()

        # Standout Slide for Language Modeling
        lang_standout = Text("Language Modeling", font_size=36)
        self.play(Write(lang_standout))
        self.wait(2)
        self.clear()

        # Task Slide for Language Modeling
        lang_task_title = Text("Language Modeling: Task", font_size=36)
        self.play(Write(lang_task_title))
        self.wait(2)

        lang_task_description = Text("Autoregressive causal language modeling with a 'decoder-only' architecture", font_size=24)
        self.play(Write(lang_task_description))
        self.wait(3)
        self.clear()

        # Results Slide for Language Modeling
        lang_results_title = Text("Language Modeling: Results", font_size=36)
        self.play(Write(lang_results_title))
        self.wait(2)

        # Assuming the language modeling image is in the assets/images directory
        lang_results_image = ImageMobject("assets/images/fineweb.png")
        lang_results_image.scale(0.5)
        self.play(FadeIn(lang_results_image))
        self.wait(3)
        self.clear()