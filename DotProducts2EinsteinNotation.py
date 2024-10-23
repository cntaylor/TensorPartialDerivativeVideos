from manim import *
import numpy as np


colors = [YELLOW, MAROON_B, PINK]

TIME_SCALE=.1 # Use this to speed things up.  1 is normal speed., <1 is faster
class Main(Scene):

    def construct(self):
        v = self.define_dot_product([[5],[3],[2]], [[1],[4],[2]], 21)
        dot_descp = Text("Two vectors as input,\n one scalar as output").next_to(v, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(dot_descp))
        self.wait(5* TIME_SCALE)        
        self.clear()
        v =self.define_dot_product([['u_1'],['u_2'],['u_3']], [['v_1'],['v_2'],['v_3']])
        sum_notation = Tex("$<u,v> = \sum_i u_i v_i$").next_to(v, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(sum_notation))
        note = Tex("Assumes $u$ and $v$ are the same length").next_to(sum_notation, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(note))
        self.wait(5 * TIME_SCALE)

    def define_dot_product(self, v1_vals, v2_vals, final_prod = None):
        # Create the vectors
        v1 = Matrix(v1_vals)
        v1.move_to(LEFT*4)

        self.play(Create(v1))
        self.wait(5 * TIME_SCALE)

        v2 = Matrix(v2_vals)
        self.play(Create(v2))
        self.wait(2 * TIME_SCALE)
    
        # Now show that we are dot producting the two vectors
        dot = Tex("$\\cdot$").next_to(v1, RIGHT).scale(1.5)
        v2.generate_target()
        v2.target.next_to(dot, RIGHT)
        v2.target.set_row_colors(*colors)

        self.play(FadeIn(dot))
        self.play(MoveToTarget(v2))

        v1_colored = v1.copy()
        v1_colored.set_row_colors(*colors)
        self.play(ReplacementTransform(v1, v1_colored))

        equals = Tex("$=$").next_to(v2, RIGHT)
        self.play(FadeIn(equals))
        self.wait(.5 * TIME_SCALE)

        #Animate the actual dot product process
        v1_entries = v1.get_entries()
        v2_entries = v2.get_entries()
        for i in range(len(v1_entries)):
            v1_entry = v1_entries[i].copy()
            v2_entry = v2_entries[i].copy()
            v1_entry.generate_target()
            v2_entry.generate_target()
            if i==0:
                v1_entry.target.next_to(equals, RIGHT)
            else:
                v1_entry.target.next_to(plus, RIGHT)
            self.play(MoveToTarget(v1_entry))
            entry_dot = Tex("$\\cdot$").next_to(v1_entry.target, RIGHT)
            self.play(Create(entry_dot))
            v2_entry.target.next_to(entry_dot, RIGHT)
            self.play(MoveToTarget(v2_entry))

            if i != len(v1_vals)-1:
                plus = Tex("$+$").next_to(v2_entry, RIGHT)
                self.play(Create(plus))

        self.wait(1* TIME_SCALE)
        
        # Show the output of the dot product, if desired
        if final_prod is not None:
            dot_prod = Tex(f'$={str(final_prod)}$').next_to(v2_entry, RIGHT)
            self.play(FadeIn(dot_prod))
            self.wait(5* TIME_SCALE)
        
        return v1 # To enable placement outside of function
        
        