from manim import *
import numpy as np

TIME_SCALE=1 # Use this to speed things up.  1 is normal speed., <1 is faster
class Main(Scene):
    def construct(self):
        # Opening quote:
        self.opening_quote()
        self.clear()
        # First, draw a derivative graph
        self.derivative_graph()
        self.clear()

    def derivative_graph(self):
        self.
    def opening_quote(self):
        line1 = Text("It is not where we start, but").scale(1).to_edge(UP).to_edge(LEFT)
        self.play(Write(line1))
        line2 = Text("where we are headed that matters most.").scale(1).next_to(line1, DOWN, aligned_edge=LEFT)
        self.play(Write(line2))
        line3 = Text("--Clark G. Gilbert").next_to(line2, DOWN*2, aligned_edge=LEFT)
        self.play(Write(line3))
        self.wait(5 * TIME_SCALE)

    def opening_quote_v1(self):
        line1 = Text("Don't worry about where you are.").scale(1).to_edge(UP).to_edge(LEFT)
        self.play(Write(line1))
        line2 = Text("Watch the first derivative.").scale(1).next_to(line1, DOWN, aligned_edge=LEFT)
        self.play(Write(line2))
        line3 = Text(" --Fred Green").next_to(line2, DOWN, aligned_edge=LEFT)
        self.play(Write(line3))
        self.wait(5 * TIME_SCALE) 

with tempconfig({"quality": "low_quality", "preview": True}):
    scene = Main()
    scene.render()
