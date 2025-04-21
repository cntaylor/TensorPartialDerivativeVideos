from manim import *
import numpy as np
import numpy.linalg as la
from functools import partial
from math import cos,sin

TIME_SCALE=1 # Use this to speed things up.  1 is normal speed., <1 is faster

#These functions end up working better if not in a class...
def surface_func(u,v):
    return np.array([u,v, np.exp(-(u**2+v**2)/16)*np.sin(np.pi/4*(2*u+v))])

def surface_grad_func(u,v):
    return np.array([-(u/8)*np.exp(-(u**2+v**2)/16)*np.sin(np.pi/4*(2*u+v)) + (np.pi/2)*np.exp(-(u**2+v**2)/16)*np.cos(np.pi/4*(2*u+v)), 
                    -(v/8)*np.exp(-(u**2+v**2)/16)*np.sin(np.pi/4*(2*u+v)) + (np.pi/4)*np.exp(-(u**2+v**2)/16)*np.cos(np.pi/4*(2*u+v))])

def simple_path(dot,alpha):
    dot.move_to(np.array([0.,-2.,0.]) + alpha * (np.array([5.,4.,4.]) - np.array([0.,-2.,0.])))

    
class Main_3D(ThreeDScene):

    start_pt = np.array([-1.,-2.])
    end_pt = np.array([5.,4.])
    arrow_length_scalar = .7
    line_params = {'color': RED, 'thickness': .04}
    axis_scale_factor = 0.9
    axis_shift = IN
    old_phi = 0.
    old_theta = 0.
    old_gamma = 0.

    def store_camera_orientation(self):
        self.old_phi = self.camera.phi
        self.old_theta = self.camera.theta
        self.old_gamma = self.camera.gamma

    def restore_camera_orientation(self):
        self.camera.phi = self.old_phi
        self.camera.theta = self.old_theta
        self.camera.gamma = self.old_gamma

    def draw_slope_line(self, deriv_func, ax, dot_loc ):
        x,point = ax.point_to_coords(dot_loc)
        
        slope = deriv_func(x)
        if abs(slope<1):
            x_dist =1
        else:
            x_dist = 2-slope
        line = Line(ax.coords_to_point(x-x_dist,point - slope*x_dist), 
                    ax.coords_to_point(x+x_dist, point + slope*x_dist), 
                    color=RED)
        slope = round(slope,2)
        deriv_txt = Tex('$\\frac{d}{dx} f(x) = '+f'{slope:.2f}$').scale(.8).next_to(line, UP)
        # dot = Dot(ax.coordinates_to_point(x,point), color=RED)
        return VGroup(line, deriv_txt)
        # return line
        
        

    def derivative_graph(self):
        ax = Axes(x_range=[-10,10])
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")
        orig_func = Tex('$f(x) = x^{1/3}$').scale(1.2).move_to(LEFT*4+UP*2.5)
        deriv_func = Tex('$\\frac{d}{dx} f(x) = \\frac{1}{3} x^{-2/3}$').scale(1.2).next_to(orig_func,DOWN, aligned_edge=LEFT)
        func = lambda x: x**(1./3) if x>=0 else -(-x)**(1./3)
        deriv = lambda x: 1./3 * x**(-2./3) if x>=0 else -(-x)**(-2./3)
        # Make graph sections so I can have the derivative slope just follow a portion of the graph
        graph1 = ax.plot(func, x_range=[-10,.5,.01], color=BLUE)
        graph2 = ax.plot(func, x_range=[.5,8.5,.01], color=BLUE)
        graph3 = ax.plot(func, x_range=[8.5, 10, .01], color=BLUE)
        self.add(ax, labels)
        self.play(Write(orig_func))
        self.play(Create(graph1))
        self.play(Create(graph2), rate_func=linear)
        self.play(Create(graph3), rate_func=linear, run_time = .5)
        # Now draw the derivative function
        self.play(Write(deriv_func))
        # And trace it along the curve
        dot = Dot().set_color(RED)
        slope_line = VMobject()
        self.add(dot, slope_line)
        # slope_line.add_updater(lambda x: x.become(Line(LEFT, dot.get_center()).set_color(RED)))
        slope_line.add_updater(lambda x: x.become(self.draw_slope_line(deriv, ax, dot.get_center())))
        self.play(MoveAlongPath(dot, graph2), run_time = 6*TIME_SCALE, rate_func=linear)
        self.wait(5 * TIME_SCALE)
        # Discussion text...
        txt_scalar = 1
        x_txt = Tex('$x$').scale(txt_scalar).move_to(RIGHT+DOWN).set_color(YELLOW)
        x_scalar_txt = Tex('is a scalar').scale(txt_scalar).next_to(x_txt, RIGHT*6.5, aligned_edge=LEFT)
        fx_txt = Tex('$f(x)$').scale(txt_scalar).next_to(x_txt, DOWN,aligned_edge=LEFT).set_color(YELLOW)
        fx_scalar_txt = Tex('is a scalar').scale(txt_scalar).next_to(x_scalar_txt, DOWN, aligned_edge=LEFT)
        result_txt = Tex('$\\Rightarrow\\frac{\\partial f(x)}{\\partial x}$').scale(txt_scalar).next_to(fx_txt, DOWN,aligned_edge=LEFT).set_color(YELLOW)
        result_scalar_txt = Tex('is a scalar').scale(txt_scalar).next_to(fx_scalar_txt, DOWN*1.5, aligned_edge=LEFT)
        self.play(Write(x_txt))
        self.play(Write(x_scalar_txt))
        self.wait(1*TIME_SCALE)
        self.play(Write(fx_txt))
        self.play(Write(fx_scalar_txt))
        self.wait(1*TIME_SCALE)
        self.play(Write(result_txt))
        self.play(Write(result_scalar_txt))
        self.wait(1*TIME_SCALE)
        txt_group = VGroup(x_txt, x_scalar_txt, fx_txt, fx_scalar_txt, result_txt, result_scalar_txt)
        # End of first example.  Add a 2nd function...
        self.play(FadeOut(txt_group), FadeOut(deriv_func), FadeOut(dot), FadeOut(slope_line))
        orig_func.generate_target()
        func_equals = Tex('$f(x)=$')
        two_func_matrix = Matrix([['x^{\\frac{1}{3}}'], 
                                  ['\\sin(\\frac{\\pi}{2}x)']],
                                  element_alignment_corner=[0.,0,0]).next_to(func_equals,RIGHT)
        # Set the colors in the vector to be equal to the plots
        mat_entries = two_func_matrix.get_entries()
        mat_entries[0].set_color(BLUE)
        mat_entries[1].set_color(GREEN)
        orig_func.target = VGroup(func_equals, two_func_matrix)
        orig_func.target.move_to(orig_func.get_center())
        func2 = lambda x: np.sin(np.pi*x/2)
        second_graph = ax.plot(func2, x_range=[-10,10,.01], color=GREEN)
        self.play(MoveToTarget(orig_func), Create(second_graph))
        self.wait(2 * TIME_SCALE)
        plots = VGroup(ax, labels, graph1, graph2, graph3, second_graph)
        plots.generate_target()
        plots.target.scale(.5).move_to(RIGHT*3,DOWN*5)
        self.play(MoveToTarget(plots))
        self.wait(3 * TIME_SCALE)   
        # Now draw the derivative function
        deriv_equals = Tex('$\\frac{\\partial f(x)}{\\partial x}=$').next_to(func_equals, DOWN*8, aligned_edge=LEFT)
        two_deriv_matrix = Matrix([['\\frac{1}{3}x^{-\\frac{2}{3}}'], 
                                  ['\\frac{\\pi}{2}\\cos(\\frac{\\pi}{2}x)']],
                                  element_alignment_corner=[0.,0,0],
                                  v_buff=1.1).next_to(deriv_equals,RIGHT)
        deriv_entries = two_deriv_matrix.get_entries()
        deriv_entries[0].set_color(BLUE)
        deriv_entries[1].set_color(GREEN)
        
        self.play(Create(deriv_equals))
        self.play(Create(two_deriv_matrix))
        self.wait(2 * TIME_SCALE)
        # Put the discussion text back in
        fx_scalar_txt = Tex('is a vector').scale(txt_scalar).next_to(x_scalar_txt, DOWN, aligned_edge=LEFT)
        result_scalar_txt = Tex('is a vector').scale(txt_scalar).next_to(fx_scalar_txt, DOWN*1.5, aligned_edge=LEFT)
        txt_group = VGroup(x_txt, x_scalar_txt, fx_txt, fx_scalar_txt, result_txt, result_scalar_txt)
        self.play(Create(txt_group))
        self.wait(2 * TIME_SCALE)

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


    def draw_proj_plane(self, proj_point, normal, extent):
        plane_origin = proj_point + normal*0.6
        v1 = np.array([1, 0, 0.])
        if np.abs(np.dot(v1, normal)) > 0.9:
             v1 = np.array([0, 1., 0])
        v1 = np.cross(v1, normal)
        v1 /= la.norm(v1)
        v2 = np.cross(normal, v1)
        v2 /= la.norm(v2)
        going_out = Surface(
            lambda u, v: plane_origin + u * v1 + v * v2,
            u_range=[-extent, extent],
            v_range=[-extent, extent],
            stroke_width=0.1,
        )
        going_out.set_fill(color = BLUE)

        return going_out

    def path_func(self, alpha):
        # Interpolation between two points
        # For now, simple linear interpolation
        # print('Got alpha', alpha,'in path_func')
        return self.start_pt + alpha * (self.end_pt - self.start_pt)

    def update_dot(self, dot, alpha):
        # path_func returns x and y.  Star breaks it out
        # surface func returns x,y, and z (in an array). Star breaks it out
        dot.move_to(surface_func(*self.path_func(alpha))*self.axis_scale_factor + self.axis_shift)

    def update_line(self, line, alpha):
        line_mid = surface_func(*self.path_func(alpha))*self.axis_scale_factor + self.axis_shift
        grad = surface_grad_func(*self.path_func(alpha))
        grad3d = np.array([grad[0], grad[1], np.dot(grad,grad)]) * self.axis_scale_factor
        # According to Gemini, the end of the arrow should be a point on the tangent plane,
        # which can be computed as z = z0 + fx(x0, y0)(x - x0) + fy(x0, y0)(y - y0)
        line_end = line_mid + grad3d*self.arrow_length_scalar
        line_start = line_mid - grad3d * self.arrow_length_scalar
        # print('alpha',alpha,'line_end',line_end,'line_start',line_start)
        line.become(
            Line3D(line_start, line_end, **self.line_params))

    def create_surface_group(self):
        # Now draw the 3D (2D surface) function
        func_surf = Surface(surface_func, v_range = [-6,6], u_range = [-6,6])
        func_surf.set_style(fill_opacity=1, stroke_color=BLUE)
        
        axes = ThreeDAxes(x_range=(-7,7,1), y_range=(-7,7,1), z_range=(-2,2,1), 
                          x_length=14, y_length=14, z_length=4)
        label = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")
        
        surf_group = (
            VGroup(axes, func_surf, label)
            .scale(self.axis_scale_factor).move_to(self.axis_shift)
        )
        return surf_group

    def run_surface_screen(self):
        self.store_camera_orientation()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

      
        # Original function in LaTeX
        orig_func = Tex('$f(\\bm{x}) = e^{-\\frac{x^2+y^2}{16}}\\sin\\left(\\frac{\\pi}{4}(2x+y)\\right);$',
                        tex_template=self.tex_template).scale(.9).move_to(LEFT*2.5+UP*3.5)
        nex_func = Tex('$\\bm{x} = \\begin{bmatrix}x \\\\ y \\end{bmatrix}$',
                       tex_template=self.tex_template).scale(.8).next_to(orig_func,RIGHT*2)
        self.add_fixed_in_frame_mobjects(orig_func,nex_func)
        # Derivatives function in LaTeX
        d_front = Tex('$\\frac{\\partial f}{\\partial \\bm{x}} = $', tex_template=self.tex_template).scale(1).next_to(orig_func,DOWN, aligned_edge=LEFT)
        d_matrix = Matrix([['-\\frac{2x}{16}e^{-\\frac{x^2+y^2}{16}}\\sin\\left(\\frac{\\pi}{4}(2x+y)\\right) + \\frac{\\pi}{2}e^{-\\frac{x^2+y^2}{16}}\\cos\\left(\\frac{\\pi}{4}(2x+y)'], 
                           ['-\\frac{2y}{16}e^{-\\frac{x^2+y^2}{16}}\\sin\\left(\\frac{\\pi}{4}(2x+y)\\right) + \\frac{\\pi}{4}e^{-\\frac{x^2+y^2}{16}}\\cos\\left(\\frac{\\pi}{4}(2x+y)\\right)']],
                           element_alignment_corner=[0.,0,0],v_buff=1.2).scale(.8).next_to(d_front,RIGHT)
        d_func = VGroup(d_front, d_matrix).next_to(orig_func,DOWN, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(d_func)
        
        surf_group = self.create_surface_group()
        self.add(surf_group)
        self.wait(3 * TIME_SCALE)

        #Now draw the gradients on the surface...
        starting_point = surface_func(self.start_pt[0],self.start_pt[1])
        # Define the moving things
        moving_dot = Dot(point=starting_point+IN, color=GREEN)
        grad1= surface_grad_func(*self.start_pt)
        grad3d = np.array([grad1[0], grad1[1], np.dot(grad1,grad1)])
        # grad3d /= np.linalg.norm(grad3d)
        # According to Gemini, the end of the arrow should be a point on the tangent plane,
        # which can be computed as z = z0 + fx(x0, y0)(x - x0) + fy(x0, y0)(y - y0)
        line_end = starting_point+grad3d*self.arrow_length_scalar
        line_begin = starting_point - grad3d * self.arrow_length_scalar
        moving_grad_line = Line3D(line_begin, line_end, **self.line_params)

        self.add(moving_grad_line, moving_dot)
        self.play(
            UpdateFromAlphaFunc(moving_dot, partial(self.update_dot)),
            UpdateFromAlphaFunc(moving_grad_line, partial(self.update_line)),
            run_time = 8 * TIME_SCALE, 

        )

        self.wait(2 * TIME_SCALE)

        self.remove(moving_dot, moving_grad_line)
        surf_group.generate_target()
        surf_group.target.scale(.5*self.axis_scale_factor).move_to(IN*1.5 + DOWN*3.1) # IN = Z axis (down), and DOWN = y-axis (left)
        self.play(MoveToTarget(surf_group))
        self.wait(1 * TIME_SCALE)
        # Discussion text...
        txt_scalar = 1
        x_txt = Tex('$x$').scale(txt_scalar).move_to(RIGHT*1.2+DOWN).set_color(YELLOW)
        x_vector_txt = Tex('is a vector').scale(txt_scalar).next_to(x_txt, RIGHT*6.5, aligned_edge=LEFT)
        fx_txt = Tex('$f(x)$').scale(txt_scalar).next_to(x_txt, DOWN,aligned_edge=LEFT).set_color(YELLOW)
        fx_scalar_txt = Tex('is a scalar').scale(txt_scalar).next_to(x_vector_txt, DOWN, aligned_edge=LEFT)
        result_txt = Tex('$\\Rightarrow\\frac{\\partial f(x)}{\\partial x}$').scale(txt_scalar).next_to(fx_txt, DOWN,aligned_edge=LEFT).set_color(YELLOW)
        result_vector_txt = Tex('is a vector').scale(txt_scalar).next_to(fx_scalar_txt, DOWN*1.5, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(x_txt)
        self.add_fixed_in_frame_mobjects(x_vector_txt)
        self.wait(1*TIME_SCALE)
        self.add_fixed_in_frame_mobjects(fx_txt)
        self.add_fixed_in_frame_mobjects(fx_scalar_txt)
        self.wait(1*TIME_SCALE)
        self.add_fixed_in_frame_mobjects(result_txt)
        self.add_fixed_in_frame_mobjects(result_vector_txt)
        self.wait(5*TIME_SCALE)
        self.clear()
        self.restore_camera_orientation()

    def animated_table_tensors(self):
        self.set_camera_orientation(0,-90*DEGREES,0)
        table_loc = LEFT*2 + UP*2
        table_scale = 0.7
        # Create the full table data
        table_data = [
            ["0", "1", "2", "3", r"\cdots"],
            ["1", "2", "3", "4", r"\cdots"],
            ["2", "3", "4", "5", r"\cdots"],
            ["3", "4", "5", "6", r"\cdots"],
            [r"\vdots", r"\vdots", r"\vdots", r"\vdots", r"\ddots"]
        ]
        
        # Define row and column labels using MathTex for LaTeX
        row_labels = [Text("0 (scalar)"), Text("1 (vector)"), Text("2 (matrix)"), Text("3D tensor"), MathTex(r"\vdots")]
        col_labels = [Text("0 (scalar)"), Text("1 (vector)"), Text("2 (matrix)"), Text("3D tensor"), MathTex(r"\cdots")]

        # Create the full table
        full_table = Table(
            table_data,
            row_labels=row_labels,
            col_labels=col_labels,
            top_left_entry=MathTex(r"\bm{x}\ \backslash\ f(\bm{x})", tex_template=self.tex_template), 
            include_outer_lines=True,
            element_to_mobject=MathTex,
            element_to_mobject_config={"tex_template": self.tex_template},
            h_buff=1.0,
            v_buff=0.6
        ).scale(.6).move_to(UP)

        # Step 1: Show just the 2x2 portion with only (0,0) entry
        initial_table = Table(
            [["0 (scalar)", ""],
             ["", ""]],
            row_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            col_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            top_left_entry=MathTex(r"\bm{x}\ \backslash\ f(\bm{x})", tex_template=self.tex_template),
            include_outer_lines=True,
            h_buff=1.0,
            v_buff=0.6
        ).scale(table_scale).move_to(table_loc)

        self.play(Create(initial_table))

        ax = Axes(x_range=[-10,10])
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")
        func = lambda x: x**(1./3) if x>=0 else -(-x)**(1./3)
        deriv = lambda x: 1./3 * x**(-2./3) if x>=0 else -(-x)**(-2./3)
        loc = RIGHT*3 + DOWN*1.5
        ax_group = VGroup(ax, labels).scale(0.6).move_to(loc)
        # Make graph sections so I can have the derivative slope just follow a portion of the graph
        graph1 = ax.plot(func, x_range=[-10,.5,.01], color=BLUE)
        graph2 = ax.plot(func, x_range=[.5,8.5,.01], color=BLUE)
        graph3 = ax.plot(func, x_range=[8.5, 10, .01], color=BLUE)
        self.add(ax_group)
        graphs = VGroup(graph1, graph2, graph3)
        self.play(Create(graphs))
        # And trace it along the curve
        x_dot = 2.5
        point = func(x_dot)
        slope = deriv(x_dot)
        if abs(slope<1):
            x_dist =1
        else:
            x_dist = 2-slope        
        dot = Dot(ax.coords_to_point(x_dot, point)).set_color(RED)
        line = Line(ax.coords_to_point(x_dot-x_dist, point - slope*x_dist), 
                    ax.coords_to_point(x_dot+x_dist,  point + slope*x_dist), 
                    color=RED)
        self.add(dot, line)
        self.wait(5 * TIME_SCALE)


         # Step 2: Show 0th row and 1st column
        temp_table_1 = Table(
            [["0 (scalar)", "1 (vector)"],
             ["", ""]],
            row_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            col_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            top_left_entry=MathTex(r"\bm{x}\ \backslash\ f(\bm{x})", tex_template=self.tex_template),
            include_outer_lines=True,
            h_buff=1.0,
            v_buff=0.6
        ).scale(table_scale).move_to(table_loc)

        self.play(Transform(initial_table, temp_table_1))
        self.wait(1 * TIME_SCALE)

        func2 = lambda x: np.sin(np.pi*x/2)
        deriv_func2 = lambda x: np.pi/2 * np.cos(np.pi*x/2)
        second_graph = ax.plot(func2, x_range=[-10,10,.01], color=GREEN)
        self.play(Create(second_graph), run_time = 2)
        point = func2(x_dot)
        slope = deriv_func2(x_dot)
        dot2 = Dot(ax.coords_to_point(x_dot, point), color=RED)
        line2 = Line(ax.coords_to_point(x_dot-x_dist, point - slope*x_dist), 
                    ax.coords_to_point(x_dot+x_dist,  point + slope*x_dist), 
                    color=RED)
        self.add(dot2, line2)
        
        self.wait(2 * TIME_SCALE)
        self.remove(ax_group, graphs, dot2, line2, second_graph, dot, line)

        # Step 3: Show 1st row and 0th column
        temp_table_2 = Table(
            [["0 (scalar)", "1 (vector)"],
             ["1 (vector)", ""]],
            row_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            col_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            top_left_entry=MathTex(r"\bm{x}\ \backslash\ f(\bm{x})",tex_template=self.tex_template),
            include_outer_lines=True,
            h_buff=1.0,
            v_buff=0.6
        ).scale(table_scale).move_to(table_loc)

        self.play(Transform(initial_table, temp_table_2))
        surf_group = self.create_surface_group().scale(0.5).move_to(DOWN+RIGHT*2)
        # Fix temp_table_2 in place so it doesn't move with the camera
        self.add_fixed_in_frame_mobjects(initial_table)

        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)
        self.play(Create(surf_group))
        self.wait(3 * TIME_SCALE)
        self.remove(surf_group)
        self.set_camera_orientation(0,-90*DEGREES,0)# set the camera back to default
        self.wait(1 * TIME_SCALE)


        # Step 4: Show 1st row and 2nd column
        temp_table_3 = Table(
            [["0 (scalar)", "1 (vector)"],
             ["1 (vector)", "2 (matrix)"]],
            row_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            col_labels=[Text("0 (scalar)"), Text("1 (vector)")],
            top_left_entry=MathTex(r"\bm{x}\ \backslash\ f(\bm{x})",tex_template=self.tex_template),
            include_outer_lines=True,
            h_buff=1.0,
            v_buff=0.6
        ).scale(table_scale).move_to(table_loc)

        self.play(Transform(initial_table, temp_table_3))
        self.wait(1)

        # Step 5: Show the full table
        self.play(Transform(initial_table, full_table))
        self.wait(2 * TIME_SCALE)

    def chain_rule_review(self):
        line1 = Tex(r'$h(x) = f(u,v) = f(g(x))$').move_to(LEFT*2.5+UP*3)
        line2 = Tex(r'$\frac{\partial h(x)}{\partial x} = \frac{\partial f(u,v)}{\partial u} \frac{\partial u}{\partial x} + \frac{\partial f(u,v)}{\partial v} \frac{\partial v}{\partial x}$').next_to(line1, DOWN, aligned_edge=LEFT)
        line3 = Tex(r'$\bm{u} = \begin{bmatrix} u \\ v \end{bmatrix}$',tex_template=self.tex_template).next_to(line2, DOWN*1.5, aligned_edge=LEFT)
        line4 = Tex(r'$h(x) = f(\bm{u}) = f(g(x))$',tex_template=self.tex_template).next_to(line3, DOWN, aligned_edge=LEFT)
        line5 = Tex(r'$\frac{\partial h(x)}{\partial x} = \sum_{i=1}^2\frac{\partial f(\bm{u})}{\partial \bm{u}_i} \frac{\partial \bm{u}_i}{\partial x}$',tex_template=self.tex_template).next_to(line4, DOWN, aligned_edge=LEFT)
        the_main_point = Text("The chain rule is just a dot product!").set_color(BLUE).move_to(DOWN*2.5)
        self.play(Write(line1))
        self.wait(0.5*TIME_SCALE)
        self.play(FadeIn(line2))
        self.wait(3 * TIME_SCALE)
        self.play(Write(line3))
        self.wait(0.5*TIME_SCALE)
        self.play(Write(line4))
        self.wait(0.5*TIME_SCALE)
        self.play(Write(line5))
        self.wait(0.5*TIME_SCALE)
        self.play(Write(the_main_point))
        self.wait(3 * TIME_SCALE)
        self.remove(line1, line2, line3, line4, line5, the_main_point)


    def projection_example1(self):
        # Formula in the top-left
        formula = Tex(r"$\bm{u}_k = \bm{d}\left(\bm{KR}(\bm{X}_k-\bm{T})\right)$",tex_template=self.tex_template).to_corner(UL)
        
        d_func = Tex(r'$d(\bm{y}) = \begin{bmatrix} \frac{\bm{y}_1}{\bm{y}_3} \\ \frac{\bm{y}_2}{\bm{y}_3}\end{bmatrix}$',tex_template=self.tex_template).next_to(formula, DOWN, aligned_edge=LEFT)
        # d_deriv = Tex(r'$\frac{\partial \bm{d}(\bm{y})_i}{\partial \bm{l}_k} = \begin{bmatrix} \frac{1}{\bm{y}_3} & 0 & -\frac{\bm{y}_1}{\bm{y}_3^2}\\ 0 & \frac{1}{\bm{y}_3} & -\frac{\bm{y}_2}{\bm{y}_3^2} \end{bmatrix}$',
        #               tex_template = self.tex_template).next_to(d_func,DOWN,aligned_edge=LEFT)
        R_scaling=0.7
        R_form = Tex(r'$\bm{R} = \begin{bmatrix} \cos\theta\cos\psi & \cos\theta\sin\psi & -\sin\theta\\ ' + 
                     r'\sin\phi\sin\theta \cos\psi - \cos\phi\sin\psi & \sin\phi\sin\theta\sin\psi + \cos\phi\cos\psi & \sin\phi\cos\theta \\' +
                     r'\cos\phi\sin\theta\cos\psi + \sin\phi\sin\psi & \cos\phi\sin\theta\sin\psi - \sin\phi\cos\psi & \cos\phi\cos\theta \end{bmatrix}$',
                     tex_template = self.tex_template).move_to(UP*2.5).scale(R_scaling)
        
        R_deriv1_pfrac = Tex(r'$\frac{\partial \bm{R}}{\partial \phi} = $', tex_template = self.tex_template, color=YELLOW)
        R_deriv1_matrix = Tex(r'$\begin{bmatrix}'+
                              r'0 & 0 & 0\\ '
                              r'\cos\phi \sin\theta \cos\psi + \sin\phi \sin\psi & \cos\phi \sin\theta \sin\psi - \sin\phi \cos\psi & \cos\phi \cos\theta\\ ' +
                              r'-\sin\phi \sin\theta \cos\psi + \cos\phi \sin\psi & -\sin\phi \sin\theta \sin\psi - \cos\phi \cos\psi & -\sin\phi \cos\theta \end{bmatrix}$',
                              tex_template = self.tex_template, color=YELLOW)
        R_deriv1_group = VGroup(R_deriv1_pfrac, R_deriv1_matrix).scale(R_scaling).arrange(RIGHT).next_to(R_form,DOWN)

        R_deriv2_pfrac = Tex(r'$\frac{\partial \bm{R}}{\partial \theta} = $', tex_template = self.tex_template, color=BLUE)
        R_deriv2_matrix = Tex(r'$\begin{bmatrix}'+
                              r'-\sin\theta \cos\psi & -\sin\theta\cos\psi & -\cos\theta\\ '+
                              r'\sin\phi\cos\theta \cos\psi  & \sin\phi\cos\theta\sin\psi  & -\sin\phi\sin\theta \\' +
                              r'\cos\phi\cos\theta\cos\psi & \cos\phi\cos\theta\sin\psi  & -\cos\phi\sin\theta \end{bmatrix}$',
                              tex_template=self.tex_template, color=BLUE)
        R_deriv2_group = VGroup(R_deriv2_pfrac, R_deriv2_matrix).scale(R_scaling).arrange(RIGHT).next_to(R_deriv1_group, DOWN)

        R_deriv3_pfrac = Tex(r'$\frac{\partial \bm{R}}{\partial \psi} = $', tex_template = self.tex_template, color=GREEN)
        R_deriv3_matrix = Tex(r'$\begin{bmatrix}'+
                              r'-\cos\theta\sin\psi & \cos\theta\cos\psi & 0 \\ '+
                              r'-\sin\phi\sin\theta \sin\psi - \cos\phi\cos\psi & \sin\phi\sin\theta\cos\psi - \cos\phi\sin\psi & 0 \\' +
                              r'-\cos\phi\sin\theta\sin\psi + \sin\phi\cos\psi & \cos\phi\sin\theta\cos\psi + \sin\phi\sin\psi & 0 \end{bmatrix}$',
                              tex_template = self.tex_template, color=GREEN)
        R_deriv3_group = VGroup(R_deriv3_pfrac, R_deriv3_matrix).scale(R_scaling).arrange(RIGHT).next_to(R_deriv2_group, DOWN)
        R_deriv_tensor = Tex(r'$\frac{\partial \bm{R}_{mn}}{\partial \bm{\beta}_j} = $', tex_template = self.tex_template).next_to(d_func,DOWN*2.5, aligned_edge=LEFT)

        d_deriv_w_t_indicies = Tex(r'$\frac{\partial \bm{d}(\bm{y})_i}{\partial \bm{y}_l} = $ \begin{blockarray}{cccc} &  & $l$ & \\ \begin{block}{c[ccc]} \multirow{2.5}{*}{$i$} & $\frac{1}{\bm{y}_3}$ & 0 & $-\frac{\bm{y}_1}{\bm{y}_3^2}$\\ & 0 & $\frac{1}{\bm{y}_3}$ & $-\frac{\bm{y}_2}{\bm{y}_3^2}$\\ \end{block} \end{blockarray}',
                      tex_template = self.tex_template).next_to(R_deriv_tensor,DOWN *2, aligned_edge=LEFT)
        full_formula = Tex(r'$\frac{\partial \bm{u}_i}{\partial \bm{\beta}_j} = \frac{\partial \bm{d}(\bm{y})_i}{\partial \bm{y}_l} \bm{K}_{lm} \frac{\partial \bm{R}_{mn}}{\partial \bm{\beta}_j} (\bm{X} - \bm{T})_n$',
                           tex_template = self.tex_template).scale(1.2).next_to(d_deriv_w_t_indicies,DOWN*2.5, aligned_edge=LEFT)
        full_formula_mult_points = \
            Tex(r'$\frac{\partial \bm{u}_{ki}}{\partial \bm{\beta}_j} = \left.\frac{\partial \bm{d}(\bm{y})_{i}}{\partial \bm{y}_l}\right|_{\bm{y}_k} \bm{K}_{lm} \frac{\partial \bm{R}_{mn}}{\partial \bm{\beta}_j} (\bm{X}_{kn} - \bm{T}_n)$',
                           tex_template = self.tex_template).scale(1.6)
        self.add_fixed_in_frame_mobjects(formula, d_func)
        

        # Projection point (center of projection)
        projection_center = np.array([-1,0,0.])
        projection_point = Sphere(radius=0.1, color=RED).move_to(projection_center)
        projection_label = Tex(r"$\bm{T}$",tex_template=self.tex_template).next_to(projection_point, DOWN + LEFT)

        # Projection plane
        plane_normal = np.array([cos(30*DEGREES), 0, -sin(30*DEGREES)])
        ortho_v1 = np.cross(plane_normal,np.array([0,0.,1]))
        ortho_v1 /= la.norm(ortho_v1)
        ortho_v2 = np.cross(plane_normal,ortho_v1)
        plane_extent = .7

        
        projection_plane = self.draw_proj_plane(projection_center, plane_normal, plane_extent)

        # 3D points (stars)
        points_3d = [
            projection_center + plane_normal*4 - ortho_v1 + 0.5*ortho_v2,
            projection_center + plane_normal*5 + ortho_v1*0.5 - 0.75* ortho_v2,
            projection_center + plane_normal * 2.5 + ortho_v1 + ortho_v2
        ]
        stars = VGroup(*[Star(5, fill_color=YELLOW, stroke_color=GOLD, outer_radius=0.2).move_to(p) for p in points_3d])
        labels_x = VGroup(*[Tex(f"$\\bm{{X}}_{i+1}$",tex_template=self.tex_template).next_to(star, RIGHT) for i, star in enumerate(stars)])

        # Projection lines
        projection_lines = VGroup(*[
            Line(projection_center, point_3d, color=BLUE_A)
            for point_3d in points_3d
        ])

        
        self.add(projection_point, projection_label, projection_plane, stars, labels_x, projection_lines)
        self.set_camera_orientation(phi=0, theta=-90.*DEGREES, gamma=0., frame_center = np.array([0,0,5.]))
        self.wait(1 * TIME_SCALE)

        # --- Animation: Change in Translation (T) ---
        new_translation = np.array([-1, 1, -0.5])
        new_projection_center = projection_center + new_translation
        new_projection_point = Sphere(radius=0.1, color=RED).move_to(new_projection_center)
        new_projection_plane = self.draw_proj_plane(new_projection_center, plane_normal, plane_extent)
        new_projection_lines = VGroup(*[
            Line(new_projection_center, point_3d, color=BLUE_A)
            for point_3d in points_3d
        ])
        new_projection_label = Tex(r"$\bm{T}'$", tex_template = self.tex_template).next_to(new_projection_point, DOWN+LEFT)

        self.play(
            Transform(projection_point, new_projection_point),
            Transform(projection_plane, new_projection_plane),
            Transform(projection_lines, new_projection_lines),
            Transform(projection_label, new_projection_label),
            run_time=3
        )
        self.wait(1)

        # --- Animation: Change in Rotation (R) ---
        rotation_angle = 20 * DEGREES

        self.play(
            Rotate(projection_plane, angle=10*DEGREES, axis=np.array([0,0,1.]), about_point=projection_center),
            run_time=2
        )
        self.play(
            Rotate(projection_plane, angle=rotation_angle, axis=plane_normal, about_point=projection_center),
            run_time=2
        )
        self.wait(1 * TIME_SCALE)
        self.remove(projection_point, projection_label, projection_plane, stars, labels_x, projection_lines)
        intro_line1 = Tex("How do we compute the derivative of $\\bm{u}$",
                          tex_template=self.tex_template, color=BLUE)
        intro_line2 = Tex('w.r.t. yaw, pitch, and roll (a representation of $\\bm{R})$?', 
                         tex_template=self.tex_template, color=BLUE).next_to(intro_line1,DOWN)
        self.play(Write(intro_line1))
        self.play(Write(intro_line2))
        self.wait(2*TIME_SCALE)
        self.clear()

        # Rotation matrix derivatives and combine into a tensor
        self.play(Write(R_form))
        self.wait(1*TIME_SCALE)
        self.play(Write(R_deriv1_group))
        self.play(Write(R_deriv2_group))
        self.play(Write(R_deriv3_group))
        self.wait(2*TIME_SCALE)
        # Merge the three groups above into one tensor
        self.remove(R_form)
        self.play(FadeIn(formula))
        self.wait(0.5*TIME_SCALE)
        self.play(FadeIn(d_func))
        # First, tell the matrices where to go:
        position_tens_eq = R_deriv_tensor.get_center()
        width_tens_eq = R_deriv_tensor.width
        R_deriv1_matrix_top_tensor = R_deriv1_matrix.copy().scale(0.8).move_to(position_tens_eq+RIGHT*width_tens_eq*.6, aligned_edge=LEFT)
        R_deriv2_matrix_mid_tensor = R_deriv2_matrix.copy().scale(0.75).move_to(position_tens_eq+RIGHT*width_tens_eq*.8+DOWN*.15, aligned_edge=LEFT)
        R_deriv3_matrix_bottom_tensor = R_deriv3_matrix.copy().scale(0.7).move_to(position_tens_eq+RIGHT*width_tens_eq+DOWN*.3, aligned_edge=LEFT)
        self.play(
            Transform(R_deriv1_group, R_deriv_tensor),
            Transform(R_deriv2_group, R_deriv_tensor),
            Transform(R_deriv3_group, R_deriv_tensor),
            Transform(R_deriv1_matrix, R_deriv1_matrix_top_tensor),
            Transform(R_deriv2_matrix, R_deriv2_matrix_mid_tensor),
            Transform(R_deriv3_matrix, R_deriv3_matrix_bottom_tensor),
            run_time=2
        )

        self.wait(1 * TIME_SCALE)
        self.play(Write(d_deriv_w_t_indicies))
        self.wait(1 * TIME_SCALE)
        self.play(Write(full_formula))
        self.wait(2*TIME_SCALE)
        self.clear()
        ff_copy = full_formula.copy()
        ff_copy.scale(1.6).to_corner(UL)
        self.play(Write(ff_copy))
        self.wait(1*TIME_SCALE)
        py_code=Text('np.einsum("il,lm,mnj,n->ij", dd_dy, K, dR_dBeta, X_minus_T)').scale(.75).next_to(ff_copy, DOWN*1.2, aligned_edge=LEFT)
        self.play(Write(py_code))
        self.wait(2*TIME_SCALE)
        full_formula_mult_points.next_to(py_code, DOWN*3, aligned_edge=LEFT)
        self.play(Write(full_formula_mult_points))
        py_code_2 = Text('np.einsum("kil,lm,mnj,kn->kij", dd_dy, K, dR_dBeta, X_minus_T)').scale(.75).next_to(full_formula_mult_points, DOWN, aligned_edge=LEFT)
        self.play(Write(py_code_2))
        self.wait(2*TIME_SCALE)

        # self.play(Transform(full_formula, full_formula_mult_points))
        # self.wait(2*TIME_SCALE)    

        # Now do an example of Python coding...
    def final_screen(self):
        review = Text("Review").set_color(BLUE).to_corner(UL)
        self.play(Write(review))
        self.wait(.2*TIME_SCALE)
        point1 = Text('1. Tensors can store high dimensional derivatives').next_to(review, DOWN, aligned_edge=LEFT)
        self.play(Write(point1))
        self.wait(.2*TIME_SCALE)
        point2 = Text('2. Tensor contraction implements the chain rule').next_to(point1, DOWN, aligned_edge=LEFT)
        self.play(Write(point2))
        self.wait(.2*TIME_SCALE)
        point3 = Text('=> Very complex derivatives can be \n\t represented using tensors').next_to(point2, DOWN*1.5, aligned_edge=LEFT)
        self.play(Write(point3))
        self.wait(2*TIME_SCALE)


    def construct(self):
        self.tex_template = TexTemplate()
        self.tex_template.add_to_preamble(r"\usepackage{bm}")
        self.tex_template.add_to_preamble(r"\usepackage{amsmath}")
        self.tex_template.add_to_preamble(r'\usepackage{blkarray}')
        self.tex_template.add_to_preamble(r'\usepackage{multirow}')
        self.tex_template.add_to_preamble(r'\usepackage{xcolor}')

        # # Opening quote:
        # self.opening_quote_v1()
        # self.clear()
        # # First, draw a derivative graph
        # self.derivative_graph()
        # self.clear()
        # # Draw a derviative graph for a (2D) surface
        # self.run_surface_screen()
        # self.animated_table_tensors()
        # self.clear()
        # self.chain_rule_review()
        # self.clear()
        self.projection_example1()
        self.clear()
        self.final_screen()


with tempconfig({"quality": "low_quality", "preview": True}):
    threeD_scene = Main_3D()
    threeD_scene.render()
