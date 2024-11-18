from manim import *
import numpy as np




TIME_SCALE=1 # Use this to speed things up.  1 is normal speed., <1 is faster
class Main(Scene):

    #TODO:  Fix all Einstein to remove commas

    def construct(self):
        # Opening quote:
        self.opening_quote()
        self.clear()
        # First, introduce tensors
        self.tensorDefine()
        self.clear()

        # Show why we need to talk about dot product
        tensor_contraction_text = Text('Tensor Contraction')
        self.play(Write(tensor_contraction_text))
        self.wait(3 * TIME_SCALE)
        tensor_contraction_text.generate_target()
        tensor_contraction_text.target = Text('Dot Product')
        self.play(MoveToTarget(tensor_contraction_text))
        self.wait(3 * TIME_SCALE)
        self.clear()

        # Show how a dot product works with a few numbers
        dot_descp = Text("Two vectors as input,\n one scalar as output").move_to(UP*2)
        self.play(FadeIn(dot_descp))
        self.wait(3* TIME_SCALE)        

        v = self.define_dot_product([[5],[3],[2]], [[1],[4],[2]], 21)
        # and give some commentary
        self.clear()
        # Now show the dot product more generally
        v =self.define_dot_product([['u_1'],['u_2'],['u_3'],['u_4']], [['v_1'],['v_2'],['v_3'],['v_4']],
                                   TIME_SCALE = 0.5, v1_pos = [-5,1,0.])
        # Show the summation notation for a dot product
        note = Tex("Assumes $u$ and $v$ are the same length").next_to(v, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(note))
        sum_notation = Tex("$<u,v> = \sum_p u_p v_p$").next_to(note, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(sum_notation))
        self.wait(5 * TIME_SCALE)
        self.clear()
        # Animate how matrix multiplcation is just a bunch of dot products
        self.matrix_mult_as_dot_products(TIME_SCALE = 0.8)
        self.clear()
        # Introduce Tensor Contraction
        self.tensorContraction()
        self.clear()
        self.einstein_notation()
        self.clear()
        # Review the whole thing
        self.backwards_review()
    
    def einstein_notation(self):
        A = self.create_tens([2,4,3], 'A').move_to([-3,2,0])
        txt_Asize = Text("A = 2x4x3").next_to(A, DOWN,aligned_edge=LEFT)
        B = self.create_tens([4,3,3], 'B').scale(.9).next_to(A, RIGHT,buff=2)
        txt_Bsize = Text("B = 4x3x3").next_to(B, DOWN, aligned_edge=LEFT)

        self.play(Create(A), Create(B), FadeIn(txt_Asize), FadeIn(txt_Bsize))
        self.wait(3 * TIME_SCALE)
        contract_opts = Table([['$C_{ijkl}=A_{pij}B_{pkl}$', '$C_{ijkl}=A_{pij}B_{kpl}$', '$C_{ijkl}=A_{pij}B_{klp}$'],
                                   ['$C_{ijkl}=A_{ipj}B_{pkl}$', '$C_{ijkl}=A_{ipj}B_{kpl}$', '$C_{ijkl}=A_{ipj}B_{klp}$'],
                                   ['$C_{ijkl}=A_{ijp}B_{pkl}$', '$C_{ijkl}=A_{ijp}B_{kpl}$', '$C_{ijkl}=A_{ijp}B_{klp}$']],
                                  row_labels=[Text('1st dim (A)'), Text('2nd dim (A)'), Text('3rd dim (A)')],
                                  col_labels=[Text('1st dim (B)'), Text('2nd dim (B)'), Text('3rd dim (B)')],
                                  top_left_entry=Text("A \\ B"),
                                  element_to_mobject=lambda x: Tex(x),
                                  include_outer_lines=True).move_to(DOWN*2.5).scale(.52)
        self.play(Create(contract_opts))
        self.wait(3 * TIME_SCALE)
        invalid1 = [(1,1), (1,2), (1,3)]
        self.cross_out_locs_in_table(contract_opts, invalid1)
        self.wait(5 * TIME_SCALE)
        invalid2 = [(2,2), (2,3), (3,1)]
        self.cross_out_locs_in_table(contract_opts, invalid2)
        self.wait(5 * TIME_SCALE)

        # Now to show multi-dimensional contraction
        self.clear()

        self.play(Create(A), Create(B), FadeIn(txt_Asize), FadeIn(txt_Bsize))
        self.wait(3 * TIME_SCALE)
        summ_1 = Tex("$C_{i,j} = \sum_{p=1}^4\sum_{q=1}^3 A_{i,p,q}B_{p,q,j}$").next_to(txt_Asize, DOWN)
        self.play(FadeIn(summ_1))
        self.wait(3 * TIME_SCALE)
        ein_1 = Tex("$C_{ij} = A_{ipq}B_{pqj}$").next_to(summ_1, DOWN)
        self.play(FadeIn(ein_1))
        self.wait(3 * TIME_SCALE)
        ein_2 = Tex("$C_{ij} = A_{ipq}B_{pjq}$").next_to(ein_1, RIGHT*4)
        self.play(FadeIn(ein_2))
        self.wait(3 * TIME_SCALE)

        
    def cross_out_locs_in_table(self,the_table,list_to_cross_out):
        for loc in list_to_cross_out:
            # Fix the fact that with row and column labels, the indicies are all off 1
            nl = (loc[0]+1, loc[1]+1) #nl = new_loc
            tmp_poly = the_table.get_cell(nl)
            vs = tmp_poly.get_vertices()
            l1 = Line(vs[0], vs[2]).set_color(RED)
            l2 = Line(vs[1], vs[3]).set_color(RED)
            self.play(Create(l1), Create(l2), run_time = .01)
        
                           
    def opening_quote(self):
        line1 = Text('Tenser, said the Tensor.').scale(1).to_edge(UP).to_edge(LEFT)
        line1[-7:].set_color(BLUE)
        self.play(Write(line1))
        # line2 = Text("Tenser, said the Tensor.").scale(1).next_to(line1, DOWN,aligned_edge=LEFT)
        # line2[-7:].set_color(BLUE)
        # self.play(Write(line2))
        # line3_and_4 = Text("Tension, apprehension,\nAnd dissension have begun.").next_to(line2, DOWN, aligned_edge=LEFT)
        # self.play(Write(line3_and_4))
        credit = Text("--Alfred Bester, The Demolished Man").next_to(line1, DOWN, aligned_edge=LEFT)
        self.play(Write(credit))
        self.wait(5 * TIME_SCALE)


    def backwards_review(self):
        review_txt = Text("Summary").to_edge(UP).scale(1.3)
        self.play(Create(review_txt))
        t1 = Text("A Tensor is a multidimensional regular array of ...").scale(.9).next_to(review_txt, DOWN).to_edge(LEFT)
        t1[1:7].set_color(BLUE)
        t1[10:26].set_color(YELLOW)
        self.play(Write(t1))
        self.wait(1*TIME_SCALE)
        t2 = Text("Using dot products, we define tensor contraction").scale(.9).next_to(t1, DOWN, aligned_edge=LEFT)
        t2[-17:].set_color(BLUE)
        t2[5:16].set_color(YELLOW)
        self.play(Write(t2))
        self.wait(1*TIME_SCALE)
        t3 = Text("We use Einstein notation to denote \n\ttensor contraction").scale(.9).next_to(t2, DOWN, aligned_edge=LEFT)
        t3[5:21].set_color(BLUE)
        t3[-17:].set_color(YELLOW)
        self.play(Write(t3))
        self.wait(1*TIME_SCALE)
        

    def tensorContraction(self):
        cont_def = Text("Tensor Contraction").to_edge(UP)
        # self.play(FadeIn(cont_def))
        # self.wait(3 * TIME_SCALE)
        # mm_text = Text("Matrix multiplication is defined as a\n particular sequence of dot products").next_to(cont_def, DOWN*2).to_edge(LEFT)
        # mm_text[:20].set_color(YELLOW)  # Matrix Multiplication
        # self.play(Write(mm_text))
        # cont_text = Text("Tensor Contraction is also defined by\n dot products").next_to(mm_text, DOWN).to_edge(LEFT)
        # cont_text[:17].set_color(BLUE) # Tensor Contraction
        # self.play(Write(cont_text))

        # self.wait(3 * TIME_SCALE)
        # self.clear()

        # mat_mult = Text('Matrix Multiplication').to_edge(UP)
        # mat_mult.set_color(YELLOW)
        # self.play(Create(mat_mult))
        # self.wait(0.5*TIME_SCALE)
        # mm_text = Text("\tIn matrix multiplication, the order of \n\tthe matrices defines which vectors to \n\tdot product together")
        # mm_text.next_to(cont_def, DOWN*2).to_edge(LEFT)
        # self.play(FadeIn(mm_text))
        
        # # mm_group = matrix mult group
        # mm_group = self.matrix_mult_as_dot_products_anim(A_pos = DOWN*2, TIME_SCALE = 0.1) 
        # self.pause(.5 * TIME_SCALE)
        # ques_text = Text("What about tensor contraction?").next_to(mm_group, DOWN).to_edge(LEFT)
        # self.play(FadeIn(ques_text))

        # self.clear()

        self.play(Create(cont_def))
        t =self.create_tens()
        v = self.create_tens([3],'v').scale(.8)
        t.next_to(cont_def, DOWN*2)
        v.next_to(t, RIGHT*2)
        self.play(Create(t), Create(v))
        #Draw three sets of vectors
        tmp = SurroundingRectangle(t[-1].get_rows()[0],color=YELLOW)
        v_rect = SurroundingRectangle(v.get_columns()[0],color=YELLOW)
        self.play(Create(tmp),Create(v_rect))
        self.pause(5 * TIME_SCALE)
        self.play(FadeOut(tmp))

        tmp = SurroundingRectangle(t[-1].get_columns()[0],color=GREEN)
        v_rect.set_color(GREEN)
        self.play(Create(tmp),Create(v_rect))
        self.pause(5 * TIME_SCALE)
        self.play(FadeOut(tmp))

        tmp =[]
        for i in range(3): 
            tmp.append (SurroundingRectangle(t[i].get_entries()[8],color=BLUE))
        tmp = VGroup(*tmp)
        v_rect.set_color(BLUE)
        self.play(Create(tmp))
        self.pause(5 * TIME_SCALE)

        q_t=Text('How do we know which dimension to "contract"?').scale(.9)
        q_t.next_to(t,DOWN)
        self.play(Write(q_t))
        self.pause(5 * TIME_SCALE)

        idx_n = Text('\tIndex notation:').next_to(q_t,DOWN,aligned_edge=LEFT)
        idx_1 = Tex('$C_{i,j} = \sum_{p=1}^3 T_{p,i,j} v_p$')
        idx_1.next_to(idx_n,RIGHT)
        self.play(Write(idx_n),Write(idx_1))
        self.pause(5 * TIME_SCALE)
        
        ein_n = Text('\tEinstein notation:').next_to(idx_n, DOWN,aligned_edge=LEFT)
        ein_1 = Tex('$C_{ij} = T_{pij} v_p$').next_to(ein_n, RIGHT)
        self.play(FadeIn(ein_n),Write(ein_1))
        self.pause(5 * TIME_SCALE)

        # Go back to index 2 contraction
        self.play(FadeOut(tmp),FadeOut(ein_1), FadeOut(idx_1))
        tmp = SurroundingRectangle(t[-1].get_columns()[0],color=GREEN)
        v_rect.set_color(GREEN)
        self.play(FadeIn(tmp))
        self.wait(2 * TIME_SCALE)

        idx_2 = Tex('$C_{i,j} = \sum_{p=1}^3 T_{i,p,j} v_p$').next_to(idx_n, RIGHT)
        ein_2 = Tex('$C_{ij} = T_{ipj} v_p$').next_to(ein_n, RIGHT)
        self.play(FadeIn(idx_2),FadeIn(ein_2))
        self.wait(5 * TIME_SCALE)
        self.play(FadeOut(tmp),FadeOut(ein_2), FadeOut(idx_2))

        # Now index 3 contraction
        tmp = SurroundingRectangle(t[-1].get_rows()[0],color=YELLOW)
        v_rect.set_color(YELLOW)
        self.play(FadeIn(tmp))
        self.wait(3 * TIME_SCALE)
        idx_3 = Tex('$C_{i,j} = \sum_{p=1}^3 T_{i,j,p} v_p$').next_to(idx_n, RIGHT)
        ein_3 = Tex('$C_{ij} = T_{ijp} v_p$').next_to(ein_n, RIGHT)
        self.play(FadeIn(idx_3), FadeIn(ein_3))
        self.wait(5 * TIME_SCALE)


    def create_tens(self, lens=[3,3,3], char = 'T'):
        # lens = lenghts of the different dimensions
        # char = character to use for the tensor
        # Retuns a VGroup with the tensor (if 3D), a Matrix with the correct stuff otherwise
        # Create the tensor
        num_dims = len(lens)
        assert num_dims > 0 and num_dims < 4, 'Can only create 1, 2, and 3D tensors'

        if num_dims == 1:
            return Matrix([[f'{char}_{i+1}'] for i in range(lens[0])])
        elif num_dims == 2:
            return Matrix([[f'{char}_{{{i+1},{j+1}}}' for j in range(lens[1])] for i in range(lens[0])])
        elif num_dims == 3:
            ts = []
            curr_scale=.6
            curr_pos = np.array([0,0,0.])
            for i in reversed(range(lens[0])):
                t = Matrix([[f'{char}_{{{i+1},{j+1},{k+1}}}' for k in range(lens[2])] for j in range(lens[1])], include_background_rectangle=True)
                t.background_rectangle.set_fill(BLACK, opacity=0.5)
                if i != lens[0]-1:
                    curr_scale *= 1.1
                    curr_pos -= np.array([1,-.5,0])
                t.move_to(curr_pos)
                t.scale(curr_scale)
                ts.append(t)
            return VGroup(*ts)
            
    def tensorDefine(self):
        """
        Animate the following:
        - A matrix is a rectangular array or table of ... with elements or entries arranged in rows and columns
        - A tensor is a multidimensional array
        - Show a vector, a matrix, and a tensor all in sequence, with the last one replacing the first two
        """
        mat_def = Text('Wikipedia:  "a matrix is a rectangular \n array or table of numbers, symbols, or\n expressions with elements or entries arranged\n in rows and columns"')
        mat_def.move_to(UP*2)
        self.play(Write(mat_def))
        self.wait(3 * TIME_SCALE)
        
        tens_def = Text('A tensor is a multidimensional array').next_to(mat_def, DOWN*3).set_color(GREEN)
        self.play(FadeIn(tens_def))
        self.wait(5 * TIME_SCALE)
        self.clear()

        # Show a vector
        v = self.create_tens([3],'v').scale(.8)
        self.play(Create(v))
        self.wait(2 * TIME_SCALE)
        m = self.create_tens([3,3],'m').scale(.8)
        self.play(ReplacementTransform(v, m))
        self.wait(2 * TIME_SCALE)
        t = self.create_tens([3,3,3],'T').scale(.8)
        t.move_to(RIGHT*.5)        
        self.play(ReplacementTransform(m, t))
        self.wait(5 * TIME_SCALE)
        t2 = self.create_tens([2,5,3],'A').scale(.8)
        t2.next_to(t,RIGHT)
        t3 = self.create_tens([4,2,2],'B').scale(.8)
        t3.next_to(t,LEFT)
        self.play(FadeIn(t2),FadeIn(t3))
        self.wait(1 * TIME_SCALE)


    def matrix_mult_as_dot_products_anim(self, A_pos=None, TIME_SCALE = 1):
        '''
        Animate how matrix multiplcation is just a bunch of dot products.  Returns a VGroup
        of the final A * B = C group
        '''
        colors = [YELLOW, BLUE, GREEN, MAROON_B, PINK, ORANGE, TEAL, PURPLE]
        # Create the first Matrix
        A = Matrix([['A_{1,1}', 'A_{1,2}', 'A_{1,3}'], ['A_{2,1}', 'A_{2,2}', 'A_{2,3}']]).scale(.8)
        if A_pos is not None:
            A.move_to(A_pos+LEFT*5)
        else:
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
                self.play(Create(tmp_entry), Create(prod_rect), run_time = TIME_SCALE if i==0 and j==0 else 0.2*TIME_SCALE)
                self.wait(0.1* TIME_SCALE)
                tmp = SurroundingRectangle(A.get_rows()[i],color=colors[i*4+j])
                tmp2 = SurroundingRectangle(B.get_columns()[j],color=colors[i*4+j])
                if i==0 and j==0:
                    self.play(Create(tmp), Create(tmp2))
                    self.wait(3 * TIME_SCALE)
                else:
                    self.play(Create(tmp), Create(tmp2), run_time = 0.4 * TIME_SCALE)
                    self.wait(0.3 * TIME_SCALE)
                self.play(FadeOut(tmp), FadeOut(tmp2), FadeOut(prod_rect))
                self.wait(.5 * TIME_SCALE)
        total_group = VGroup(A, B, equals, Prod)
        return total_group

    def matrix_mult_as_dot_products(self, TIME_SCALE = 1):
        total_group = self.matrix_mult_as_dot_products_anim()
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
        Einstein_text = Text("Einstein Notation").scale(1.2).next_to(index_two, DOWN)
        self.play(FadeIn(Einstein_text))
        self.wait(2*TIME_SCALE)
        ein_not = Tex('$C_{ij} = A_{ip} B_{pj}$').next_to(Einstein_text, DOWN)
        self.play(FadeIn(ein_not))
        self.wait(2*TIME_SCALE) 
   

    def define_dot_product(self, v1_vals, v2_vals, final_prod = None, 
                           TIME_SCALE = 1, v1_pos = LEFT*4):
        colors = [YELLOW, MAROON_B, PINK, BLUE]

        # Create the vectors
        v1 = Matrix(v1_vals)
        v1.move_to(v1_pos)

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
    
with tempconfig({"quality": "high_quality", "preview": True}):
    scene = Main()
    scene.render()
        
        