from manim import *
from manim_slides import Slide
from utils.colors import *
from utils.latex_macros import *
from pathlib import Path
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

class Main(Slide):
    def pdf_to_image_mobject(self, pdf_path, scale=0.6, dpi=200):
        """Convert PDF to ImageMobject via numpy array"""
        pages = convert_from_path(pdf_path, dpi=dpi)
        img_pil = pages[0]
        img_np = np.array(img_pil).astype(np.uint8)
        return ImageMobject(img_np).scale(scale)

    def construct(self):
        # Title Slide
        title = Text(
            "Disentangling and Integrating Relational and\nSensory Information in Transformer Architectures",
            font_size=28,
            color=WHITE,
            t2c={"Relational": RELATIONAL_COLOR, "Sensory": SENSORY_COLOR}
        ).move_to(UP * 1.5)

        authors = Text(
            "Awni Altabaa, John Lafferty",
            font_size=22,
            color=GRAY_B,
            slant=ITALIC
        ).move_to(UP * 0.5)

        institution = Text(
            "Yale University",
            font_size=20,
            color=GRAY_C
        ).move_to(DOWN * 0.5)

        paper_info = Text(
            "arXiv:2405.16727, ICML '25",
            font_size=18,
            color=GRAY_C
        ).move_to(DOWN * 1.2)

        self.play(
            FadeIn(title, shift=DOWN),
            FadeIn(authors, shift=DOWN),
            FadeIn(institution, shift=DOWN),
            FadeIn(paper_info, shift=DOWN),
            run_time=2
        )
        self.next_slide()

        # Big Picture Slide 1
        self.play(FadeOut(title), FadeOut(authors), FadeOut(institution), FadeOut(paper_info))

        section_title = Text(
            "Big Picture: Why should we care about \"relational reasoning\"?",
            font_size=24,
            color=HIGHLIGHT_COLOR
        ).to_edge(UP, buff=0.5)

        slide_title = Text("The Fundamental Principles of Intelligence", font_size=32, color=WHITE).center()

        hypothesis = VGroup(
            Text("Hypothesis 0:", font_size=24, color=ALERT_COLOR),
            Text(
                "Human & animal intelligence can be explained by a few\ncore principles (rather than an encyclopedic list of heuristics)",
                font_size=20,
                color=WHITE
            )
        ).arrange(RIGHT, buff=0.5).move_to(UP * 0.5)

        goal_text = VGroup(
            Text("Suggests the following goal:", font_size=24, color=ALERT_COLOR),
            Text(
                "Study & uncover the inductive biases that humans &\nanimals exploit to understand intelligence generally\nand inform design of AI",
                font_size=20,
                color=WHITE
            )
        ).arrange(RIGHT, buff=0.5).move_to(DOWN * 1)

        self.play(Write(section_title))
        self.play(Write(slide_title))
        self.next_slide()

        self.play(FadeIn(hypothesis))
        self.next_slide()

        self.play(FadeIn(goal_text))
        self.next_slide()

        # Big Picture Slide 2
        self.play(FadeOut(slide_title), FadeOut(hypothesis), FadeOut(goal_text))

        dl_systems = Text(
            "Deep learning systems themselves exploit several key\ninductive biases that underly their empirical success",
            font_size=22,
            color=WHITE
        ).move_to(UP * 1.5)

        ai_goal = VGroup(
            Text("Goal of AI Research:", font_size=24, color=ALERT_COLOR),
            Text(
                "Uncover a core set of inductive biases for DL that enable\ndata-efficient learning and reasoning over wide range\nof tasks and modalities",
                font_size=20,
                color=WHITE
            )
        ).arrange(RIGHT, buff=0.5).center()

        hypothesis1 = VGroup(
            Text("Hypothesis 1:", font_size=24, color=ALERT_COLOR),
            Text(
                "Relational reasoning is one of these fundamental\nprinciples of intelligence",
                font_size=20,
                color=WHITE,
                t2c={"Relational reasoning": RELATIONAL_COLOR}
            )
        ).arrange(RIGHT, buff=0.5).move_to(DOWN * 1.5)

        self.play(FadeIn(dl_systems))
        self.next_slide()

        self.play(FadeIn(ai_goal))
        self.next_slide()

        self.play(FadeIn(hypothesis1))
        self.next_slide()

        # Standout slide: What is relational reasoning?
        self.play(FadeOut(section_title), FadeOut(dl_systems), FadeOut(ai_goal), FadeOut(hypothesis1))

        standout_question = Text(
            "First: what is \"relational reasoning\"?",
            font_size=36,
            color=HIGHLIGHT_COLOR,
            weight=BOLD
        ).center()

        self.play(Write(standout_question))
        self.next_slide()

        # Definition of relational reasoning
        self.play(FadeOut(standout_question))

        definition_title = Text("First: What is \"relational reasoning\"?", font_size=28, color=HIGHLIGHT_COLOR).to_edge(UP)

        definition_points = VGroup(
            Text(
                "Reasoning about relationships between objects and\nhow they interact in a given context/scene",
                font_size=22,
                color=WHITE,
                t2c={"relationships": ALERT_COLOR}
            ),
            Text(
                "Perform comparisons under different attributes or features,\nat multiple levels of abstraction",
                font_size=22,
                color=WHITE,
                t2c={"comparisons": ALERT_COLOR}
            ),
            Text(
                "Beyond recognizing individual objects by sensory pattern recognition;\nrequires higher-order relationships",
                font_size=22,
                color=WHITE,
                t2c={"higher-order": ALERT_COLOR}
            ),
            VGroup(
                Text("Clue to its importance:", font_size=22, color=ALERT_COLOR),
                Text(
                    "Humans have a natural ability (and a preference)\nto do relational reasoning",
                    font_size=22,
                    color=WHITE
                )
            ).arrange(RIGHT, buff=0.3)
        ).arrange(DOWN, buff=0.8, aligned_edge=LEFT).center()

        self.play(Write(definition_title))
        self.next_slide()

        for point in definition_points:
            self.play(FadeIn(point))
            self.next_slide()

        # Examples introduction
        self.play(FadeOut(definition_title), FadeOut(definition_points))

        examples_intro = Text(
            "Let's walk through a couple simple illustrative\nexamples of relational tasks",
            font_size=32,
            color=HIGHLIGHT_COLOR,
            weight=BOLD
        ).center()

        self.play(Write(examples_intro))
        self.next_slide()

        # SET Card Game Example
        self.play(FadeOut(examples_intro))

        set_title = Text("Example: SET! Card Game", font_size=32, color=HIGHLIGHT_COLOR).to_edge(UP)

        # SET game visualization with actual images
        import os;
        print(os.getcwd())
        required_files = [
            "assets/figs/set_1.png",
            "assets/figs/set_2.png",
        ]

        for file in required_files:
            if not Path(file).is_file():
                raise FileNotFoundError(f"Required file not found: {file}")
        set_image1 = ImageMobject("assets/figs/set_1.png").scale(0.8).move_to(LEFT * 3)
        set_image2 = ImageMobject("assets/figs/set_2.png").scale(0.8).move_to(RIGHT * 3)

        set_desc = Text(
            "Find three cards where each attribute is\neither all the same or all different",
            font_size=20,
            color=WHITE,
            t2c={"all the same or all different": RELATIONAL_COLOR}
        ).move_to(DOWN * 2.5)

        self.play(Write(set_title))
        self.play(FadeIn(set_image1), FadeIn(set_image2))
        self.play(FadeIn(set_desc))
        self.next_slide()

        # Relational Games Example
        self.play(FadeOut(set_title), FadeOut(set_image1), FadeOut(set_image2), FadeOut(set_desc))

        relgames_title = Text("Example: Relational Games (Shanahan et al. 2020)", font_size=28, color=HIGHLIGHT_COLOR).to_edge(UP)

        # Relational games visualization with actual image
        relgames_image = ImageMobject("assets/figs/relgames_example.png").scale(0.6).center()

        relgames_desc = Text(
            "A Visual Relational Reasoning Task:\ndetermine whether a particular relation holds or not",
            font_size=22,
            color=WHITE,
            t2c={"Visual Relational Reasoning Task": ALERT_COLOR}
        ).to_edge(DOWN, buff=1)

        self.play(Write(relgames_title))
        self.play(FadeIn(relgames_image))
        self.play(FadeIn(relgames_desc))
        self.next_slide()

        # Why care about relational reasoning - standout
        self.play(FadeOut(relgames_title), FadeOut(relgames_image), FadeOut(relgames_desc))

        why_care_standout = Text(
            "Returning to our original question:\n\nWhy should we care about relational reasoning?",
            font_size=32,
            color=HIGHLIGHT_COLOR,
            weight=BOLD
        ).center()

        self.play(Write(why_care_standout))
        self.next_slide()

        # Why care - detailed answer
        self.play(FadeOut(why_care_standout))

        why_care_title = Text("Why should we care about \"relational reasoning\"?", font_size=28, color=HIGHLIGHT_COLOR).to_edge(UP)

        cornerstone = Text(
            "A cornerstone of human intelligence",
            font_size=24,
            color=WHITE
        ).move_to(UP * 1.5)

        capabilities = Text("Underlies capabilities for", font_size=22, color=WHITE).move_to(UP * 0.5)

        capability_list = VGroup(
            Text("• analogy", font_size=20, color=ALERT_COLOR),
            Text("• abstraction", font_size=20, color=ALERT_COLOR),
            Text("• generalization", font_size=20, color=ALERT_COLOR)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(ORIGIN)

        key_insight = Text(
            "By relating new inputs to previously-seen stimuli, we form\nanalogies and abstractions that allow us to\nsystematically generalize.",
            font_size=22,
            color=WHITE,
            t2c={
                "analogies": RELATIONAL_COLOR,
                "abstractions": RELATIONAL_COLOR,
                "systematically generalize": ALERT_COLOR
            }
        ).move_to(DOWN * 1.5)

        self.play(Write(why_care_title))
        self.play(FadeIn(cornerstone))
        self.next_slide()

        self.play(FadeIn(capabilities))
        self.play(FadeIn(capability_list))
        self.next_slide()

        self.play(FadeIn(key_insight))
        self.next_slide()

        # Goyal & Bengio quote - standout
        self.play(FadeOut(why_care_title), FadeOut(cornerstone), FadeOut(capabilities),
                 FadeOut(capability_list), FadeOut(key_insight))

        quote = Text(
            "\"In the limit, relational reasoning yields universal\ninductive generalization from a finite and often very\nsmall set of observed cases to a potentially infinite\nset of novel instances.\"\n\n— Goyal & Bengio (2022)",
            font_size=24,
            color=WHITE,
            slant=ITALIC,
            t2c={"relational reasoning": RELATIONAL_COLOR, "universal inductive generalization": ALERT_COLOR}
        ).center()

        self.play(Write(quote))
        self.next_slide()

        # Goal statement - standout
        self.play(FadeOut(quote))

        goal_statement = Text(
            "We'd like to take a step towards this\ncentral goal of AI research",
            font_size=36,
            color=HIGHLIGHT_COLOR,
            weight=BOLD
        ).center()

        self.play(Write(goal_statement))
        self.next_slide()

        # Outline slide
        self.play(FadeOut(goal_statement))

        outline_title = Text("Outline of Remainder of Talk", font_size=32, color=HIGHLIGHT_COLOR).to_edge(UP)

        outline_points = VGroup(
            Text("1. Transformers: The Sensory and the Relational", font_size=24, color=WHITE),
            Text("2. Dual Attention Transformer (DAT) Architecture", font_size=24, color=WHITE),
            Text("3. Empirical Investigation", font_size=24, color=WHITE),
            Text("4. Concluding Remarks", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.8, aligned_edge=LEFT).center()

        self.play(Write(outline_title))
        self.next_slide()

        for point in outline_points:
            self.play(FadeIn(point))
            self.next_slide()

        # Transition to Transformers
        self.play(FadeOut(outline_title), FadeOut(outline_points))

        transformers_title = Text(
            "Transformers: The Sensory and the Relational",
            font_size=32,
            color=HIGHLIGHT_COLOR,
            t2c={"Sensory": SENSORY_COLOR, "Relational": RELATIONAL_COLOR}
        ).center()

        self.play(Write(transformers_title))
        self.next_slide()

        # Dual Attention Transformer
        self.play(FadeOut(transformers_title))

        dat_title = Text(
            "Dual Attention Transformer Architecture",
            font_size=32,
            color=HIGHLIGHT_COLOR
        ).to_edge(UP)

        # Show architecture diagram
        architecture_image = self.pdf_to_image_mobject("assets/figs/attn_fig_combined.pdf", scale=0.7).center()

        self.play(Write(dat_title))
        self.play(FadeIn(architecture_image))
        self.next_slide()

        # Mathematical formulation
        self.play(FadeOut(architecture_image))

        math_title = Text("Dual Attention Formulation", font_size=28, color=HIGHLIGHT_COLOR).to_edge(UP)

        # Key equations
        sensory_eq = MathTex(
            r"\text{Sensory Attention: } e_i = \text{Concat}(e_i^{(1)}, \ldots, e_i^{(n_h^{sa})}) W_o^{sa}",
            font_size=20
        ).move_to(UP * 1.5)

        relational_eq = MathTex(
            r"\text{Relational Attention: } a_i = \text{Concat}(a_i^{(1)}, \ldots, a_i^{(n_h^{ra})}) W_o^{ra}",
            font_size=20
        ).move_to(UP * 0.5)

        relations_eq = MathTex(
            r"\mathbf{r}_{ij} = \langle x_i W_q^{rel}, x_j W_k^{rel} \rangle",
            font_size=20
        ).move_to(DOWN * 0.5)

        output_eq = MathTex(
            r"\text{Output: } \text{Concat}(e_i, a_i)",
            font_size=20
        ).move_to(DOWN * 1.5)

        self.play(Transform(dat_title, math_title))
        self.play(Write(sensory_eq))
        self.play(Write(relational_eq))
        self.play(Write(relations_eq))
        self.play(Write(output_eq))
        self.next_slide()

        # Clear math for next section
        self.play(
            FadeOut(math_title), FadeOut(sensory_eq),
            FadeOut(relational_eq), FadeOut(relations_eq), FadeOut(output_eq)
        )

        # Empirical Investigation
        self.play(FadeOut(dat_title))

        empirical_title = Text("Empirical Investigation", font_size=36, color=HIGHLIGHT_COLOR).to_edge(UP)

        # Relational Games Results
        relgames_results_title = Text(
            "Relational Games Results",
            font_size=28,
            color=RELATIONAL_COLOR
        ).move_to(UP * 2)

        # Load the image file directly and convert to numpy uint8 array
        relgames_plot = self.pdf_to_image_mobject("assets/figs/experiments/relgames_learning_curves.pdf", scale=0.6).center()

        relgames_caption = Text(
            "DAT outperforms standard Transformers on relational reasoning tasks",
            font_size=20,
            color=WHITE,
            t2c={"DAT outperforms": ALERT_COLOR}
        ).move_to(DOWN * 2.5)

        self.play(Write(empirical_title))
        self.play(Write(relgames_results_title))
        self.play(FadeIn(relgames_plot))
        self.play(FadeIn(relgames_caption))
        self.next_slide()

        # ImageNet Results
        self.play(
            FadeOut(relgames_results_title),
            FadeOut(relgames_plot),
            FadeOut(relgames_caption)
        )

        imagenet_results_title = Text(
            "ImageNet Classification Results",
            font_size=28,
            color=SENSORY_COLOR
        ).move_to(UP * 2)

        # Convert PDF to numpy array for ImageMobject
        imagenet_plot = self.pdf_to_image_mobject("assets/figs/experiments/imagenet_acc_curves.pdf", scale=0.6).center()

        imagenet_caption = Text(
            "Competitive performance on sensory tasks while maintaining relational capabilities",
            font_size=20,
            color=WHITE,
            t2c={"Competitive performance": ALERT_COLOR}
        ).move_to(DOWN * 2.5)

        self.play(Write(imagenet_results_title))
        self.play(FadeIn(imagenet_plot))
        self.play(FadeIn(imagenet_caption))
        self.next_slide()

        # Clear for conclusion
        self.play(
            FadeOut(empirical_title),
            FadeOut(imagenet_results_title),
            FadeOut(imagenet_plot),
            FadeOut(imagenet_caption)
        )

        # Conclusion Slide
        self.play(FadeOut(empirical_title))

        conclusion_title = Text("Concluding Remarks", font_size=36, color=HIGHLIGHT_COLOR).center()

        conclusion_points = VGroup(
            Text("• Disentangled relational and sensory processing", font_size=22, color=WHITE),
            Text("• Improved performance on relational reasoning tasks", font_size=22, color=WHITE),
            Text("• Step towards universal neural architectures", font_size=22, color=WHITE)
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(DOWN * 0.5)

        self.play(Write(conclusion_title))
        self.next_slide()

        for point in conclusion_points:
            self.play(FadeIn(point))
            self.next_slide()

        # Thank You Slide
        self.play(FadeOut(conclusion_title), FadeOut(conclusion_points))

        thank_you = Text("Thank You", font_size=48, color=HIGHLIGHT_COLOR).center()
        questions = Text("Questions?", font_size=32, color=WHITE).move_to(DOWN * 1)

        self.play(Write(thank_you))
        self.play(FadeIn(questions))
        self.next_slide()