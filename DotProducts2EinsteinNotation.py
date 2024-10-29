from manim import *
import numpy as np




TIME_SCALE=.1 # Use this to speed things up.  1 is normal speed., <1 is faster
class Main(Scene):

    def construct(self):
        # # First, show how a dot product works with a few numbers
        # v = self.define_dot_product([[5],[3],[2]], [[1],[4],[2]], 21)
        # # and give some commentary
        # dot_descp = Text("Two vectors as input,\n one scalar as output").next_to(v, DOWN, aligned_edge=LEFT)
        # self.play(FadeIn(dot_descp))
        # self.wait(5* TIME_SCALE)        
        # self.clear()
        # # Now show the dot product more generally
        # v =self.define_dot_product([['u_1'],['u_2'],['u_3']], [['v_1'],['v_2'],['v_3']])
        # # Show the summation notation for a dot product
        # sum_notation = Tex("$<u,v> = \sum_i u_i v_i$").next_to(v, DOWN, aligned_edge=LEFT)
        # self.play(FadeIn(sum_notation))
        # note = Tex("Assumes $u$ and $v$ are the same length").next_to(sum_notation, DOWN, aligned_edge=LEFT)
        # self.play(FadeIn(note))
        # self.wait(5 * TIME_SCALE)
        # self.clear()
        # Animate how matrix multiplcation is just a bunch of dot products
        # self.matrix_mult_as_dot_products()
        # self.clear()
        # Now to introduce tensors
        self.tensorStuff()

    def create_tens(self, *kwargs):
        num_dims = len(kwargs)
        assert num_dims > 0 and num_dims < 4, 'Can only create 1, 2, and 3D tensors'

        if num_dims == 1:
            return Matrix([[f'T_{i+1}'] for i in range(kwargs[0])])
        elif num_dims == 2:
            return Matrix([[f'T_{i+1,j+1}' for j in range(kwargs[1])] for i in range(kwargs[0])])
        elif num_dims == 3:
            ts = []
            for i in range(kwargs[0]):
                ts.append(Matrix([[f'T_{i+1,j+1,k+1}' for k in range(kwargs[2])] for j in range(kwargs[1])]))
                if i != 0:
                    ts[i].next_to(ts[i-1], DOWN,RIGHT)
            return VGroup(*ts)
            
    def tensorStuff(self):
        mat_def = Text('Wikipedia says:\n A matrix is a rectangular array or table of ...\n with elements or entries arranged in rows and columns')
        self.play(FadeIn(mat_def))
        self.wait(3 * TIME_SCALE)
        tens_def = Text('A tensor is a multidimensional array').next_to(mat_def, DOWN)
        self.play(FadeIn(tens_def))
        self.wait(3 * TIME_SCALE)
        self.clear()

        # Show a vector
        v = self.create_tens(3).scale(.8)
        self.play(Create(v))
        m = self.create_tens(3,3).scale(.8)
        self.play(ReplacementTransform(v, m))
        self.wait(1 * TIME_SCALE)
        t1 = Matrix([['T_{1,1,1}', 'T_{1,1,2}', 'T_{1,1,3}'], 
                    ['T_{1,2,1}', 'T_{1,2,2}', 'T_{1,2,3}'], 
                    ['T_{1,3,1}', 'T_{1,3,2}', 'T_{1,3,3}']]).scale(.8)
        t2 = Matrix([['T_{2,1,1}', 'T_{2,1,2}', 'T_{2,1,3}'], 
                    ['T_{2,2,1}', 'T_{2,2,2}', 'T_{2,2,3}'], 
                    ['T_{2,3,1}', 'T_{2,3,2}', 'T_{2,3,3}']]).scale(.8)        
        t3 = Matrix([['T_{3,1,1}', 'T_{3,1,2}', 'T_{3,1,3}'], 
                    ['T_{3,2,1}', 'T_{3,2,2}', 'T_{3,2,3}'], 
                    ['T_{3,3,1}', 'T_{3,3,2}', 'T_{3,3,3}']]).scale(.8)
        t2.move_to(t1).move_to(RIGHT).move_to(DOWN)
        t3.move_to(t1).move_to(RIGHT*2).move_to(DOWN*2)
        t_group = VGroup(t1, t2, t3)        
        self.play(ReplacementTransform(m, t_group))
        self.wait(5 * TIME_SCALE)




    def matrix_mult_as_dot_products(self):
        colors = [YELLOW, BLUE, GREEN, MAROON_B, PINK, ORANGE, TEAL, PURPLE]
        # Create the first Matrix
        A = Matrix([['A_{1,1}', 'A_{1,2}', 'A_{1,3}'], ['A_{2,1}', 'A_{2,2}', 'A_{2,3}']]).scale(.8)
        A.move_to(LEFT*5)
        B = Matrix([['B_{1,1}', 'B_{1,2}', 'B_{1,3}', 'B_{1,4}'], 
                    ['B_{2,1}', 'B_{2,2}', 'B_{2,3}', 'B_{2,4}'], 
                    ['B_{3,1}', 'B_{3,2}', 'B_{3,3}', 'B_{3,4}']]).scale(.8)
        B.next_to(A, RIGHT)
        m_group=VGroup(A, B)
        self.play(Create(m_group))
        self.wait(5 * TIME_SCALE)
        equals = Tex("$=$").next_to(B, RIGHT).scale(1.1)
        Prod = Matrix([['C_{1,1}', 'C_{1,2}', 'C_{1,3}', 'C_{1,4}' ], ['C_{2,1}', 'C_{2,2}', 'C_{2,3}', 'C_{2,4}']]).next_to(equals, RIGHT,buff=0).scale(0.8)
        Prod_brackets = Prod.get_brackets()
        self.play(FadeIn(equals), Create(Prod_brackets))
        self.wait(5 * TIME_SCALE)
        for i in range(2): # rows of prod
            for j in range(4): # cols of prod
                tmp_entry = Prod.get_entries()[i*4+j]
                prod_rect = SurroundingRectangle(tmp_entry, color=colors[i*4+j])
                self.play(Create(tmp_entry), Create(prod_rect), run_time = 1 if i==0 and j==0 else 0.2)
                self.wait(0.5* TIME_SCALE)
                tmp = SurroundingRectangle(A.get_rows()[i],color=colors[i*4+j])
                tmp2 = SurroundingRectangle(B.get_columns()[j],color=colors[i*4+j])
                if i==0 and j==0:
                    self.play(Create(tmp), Create(tmp2))
                    self.wait(3 * TIME_SCALE)
                else:
                    self.play(Create(tmp), Create(tmp2), run_time = 0.4)
                    self.wait(0.3 * TIME_SCALE)
                self.play(FadeOut(tmp), FadeOut(tmp2), FadeOut(prod_rect))
                self.wait(.5 * TIME_SCALE)
        total_group = VGroup(A, B, equals, Prod)
        total_group.move_to(UP*2)
        self.wait(1 * TIME_SCALE)
        
        # Start introducing index notation!
        index_one = Tex('$C_{i,j} = A_{i,1} B_{1,j} + A_{i,2} B_{2,j} + A_{i,3} B_{3,j}$').next_to(total_group, DOWN)
        self.play(FadeIn(index_one))
        self.wait(3 * TIME_SCALE)
        index_two = Tex('$C_{i,j} = \\sum_{p=1}^3 A_{i,p} B_{p,j}$').next_to(index_one, DOWN)
        self.play(FadeIn(index_two))
        self.wait(3 * TIME_SCALE)

        self.clear()
        index_two.move_to(UP)
        self.play(FadeIn(index_two))
        self.wait(3*TIME_SCALE)
        Einstein_text = Text("Einstein Notation").scale(1.5).next_to(index_two, DOWN)
        self.play(FadeIn(Einstein_text))
        self.wait(2*TIME_SCALE)
        ein_not = Tex('$C_{i,j} = A_{i,p} B_{p,j}$').next_to(Einstein_text, DOWN)
        self.play(FadeIn(ein_not))
        self.wait(2*TIME_SCALE) 
   

    def define_dot_product(self, v1_vals, v2_vals, final_prod = None):
        colors = [YELLOW, MAROON_B, PINK]

        # Create the vectors
        v1 = Matrix(v1_vals)
        v1.move_to(LEFT*4)

        v2 = Matrix(v2_vals)
        self.play(Create(v1), Create(v2))
        self.wait(2 * TIME_SCALE)
    
        # Now show that we are dot producting the two vectors
        dot = Tex("$\\cdot$").next_to(v1, RIGHT).scale(1.5)
        v2.generate_target()
        v2.target.next_to(dot, RIGHT)
        v2.target.set_row_colors(*colors)

        v1_colored = v1.copy()
        v1_colored.set_row_colors(*colors)
        self.play(FadeIn(dot), MoveToTarget(v2), ReplacementTransform(v1, v1_colored))

        

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
            entry_dot = Tex("$\\cdot$").next_to(v1_entry.target, RIGHT)
            v2_entry.target.next_to(entry_dot, RIGHT)
            self.play(MoveToTarget(v1_entry), MoveToTarget(v2_entry), Create(entry_dot))
            
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
    
with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = Main()
    scene.render()
        
        