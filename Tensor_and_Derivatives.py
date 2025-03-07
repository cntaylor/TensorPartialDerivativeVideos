from manim import *
import numpy as np
from functools import partial

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

    def path_func(self, alpha):
        # Interpolation between two points
        # For now, simple linear interpolation
        # print('Got alpha', alpha,'in path_func')
        return self.start_pt + alpha * (self.end_pt - self.start_pt)

    def update_dot(self, dot, alpha):
        # path_func returns x and y.  Star breaks it out
        # surface func returns x,y, and z (in an array). Star breaks it out
        dot.move_to(surface_func(*self.path_func(alpha)))

    def update_line(self, line, alpha):
        line_mid = surface_func(*self.path_func(alpha))
        grad = surface_grad_func(*self.path_func(alpha))
        grad3d = np.array([grad[0], grad[1], np.dot(grad,grad)])
        # According to Gemini, the end of the arrow should be a point on the tangent plane,
        # which can be computed as z = z0 + fx(x0, y0)(x - x0) + fy(x0, y0)(y - y0)
        line_end = line_mid + grad3d*self.arrow_length_scalar
        line_start = line_mid - grad3d * self.arrow_length_scalar
        # print('alpha',alpha,'line_end',line_end,'line_start',line_start)
        line.become(
            Line3D(line_start, line_end, **self.line_params))



    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{bm}")
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        
        # Original function in LaTeX
        orig_func = Tex('$f(\\bm{x}) = e^{-\\frac{x^2+y^2}{16}}\\sin\\left(\\frac{\\pi}{4}(2x+y)\\right);$',
                        tex_template=myTemplate).scale(1.2).move_to(LEFT*2.5+UP*3)
        nex_func = Tex('$\\bm{x} = \\begin{bmatrix}x \\\\ y \\end{bmatrix}$',
                       tex_template=myTemplate).next_to(orig_func,RIGHT*2)
        self.add_fixed_in_frame_mobjects(orig_func,nex_func)
        # # Derivatives function in LaTeX
        # d_front = Tex('$\\frac{\\partial f}{\\partial \\bm{x}} = $', tex_template=myTemplate).scale(1.2).next_to(orig_func,DOWN, aligned_edge=LEFT)
        # d_matrix = Matrix([['-\\frac{2x}{16}e^{-\\frac{x^2+y^2}{16}}\\sin\\left(\\frac{\\pi}{4}(2x+y)\\right) + \\frac{\\pi}{2}e^{-\\frac{x^2+y^2}{16}}\\cos\\left(\\frac{\\pi}{4}(2x+y)'], 
        #                    ['-\\frac{2y}{16}e^{-\\frac{x^2+y^2}{16}}\\sin\\left(\\frac{\\pi}{4}(2x+y)\\right) + \\frac{\\pi}{4}e^{-\\frac{x^2+y^2}{16}}\\cos\\left(\\frac{\\pi}{4}(2x+y)\\right)']],
        #                    element_alignment_corner=[0.,0,0]).next_to(d_front,RIGHT)
        # d_func = VGroup(d_front, d_matrix).next_to(orig_func,DOWN, aligned_edge=LEFT)
        # self.add_fixed_in_frame_mobjects(d_func)
        func_surf = Surface(surface_func, v_range = [-6,6], u_range = [-6,6])
        func_surf.set_style(fill_opacity=1, stroke_color=BLUE)
        
        axes = ThreeDAxes(x_range=(-7,7,1), y_range=(-7,7,1), z_range=(-2,2,1), 
                          x_length=14, y_length=14, z_length=4)
        self.add(axes, func_surf)
        label = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")
        self.add(label)
        self.wait(3 * TIME_SCALE)

        #Now draw the gradients on the surface...
        starting_point = surface_func(self.start_pt[0],self.start_pt[1])
        # Define the moving things
        moving_dot = Dot(point=starting_point, color=GREEN)
        grad1= surface_grad_func(*self.start_pt)
        grad3d = np.array([grad1[0], grad1[1], np.dot(grad1,grad1)])
        # grad3d /= np.linalg.norm(grad3d)
        # According to Gemini, the end of the arrow should be a point on the tangent plane,
        # which can be computed as z = z0 + fx(x0, y0)(x - x0) + fy(x0, y0)(y - y0)
        line_end = starting_point+grad3d*self.arrow_length_scalar
        line_begin = starting_point - grad3d * self.arrow_length_scalar
        moving_grad_line = Line3D(line_begin, line_end, **self.line_params)

        self.add(moving_grad_line, moving_dot)
        self.wait(1)
        # self.play(Create(moving_grad_line), run_time = 3* TIME_SCALE)
        self.play(
            UpdateFromAlphaFunc(moving_dot, partial(self.update_dot)),
            UpdateFromAlphaFunc(moving_grad_line, partial(self.update_line)),
            run_time = 6 * TIME_SCALE, 

        )

        self.wait(5 * TIME_SCALE)
        # Discussion text...




class Main(Scene):
    def construct(self):
        # Opening quote:
        self.opening_quote_v1()
        self.clear()
        # First, draw a derivative graph
        self.derivative_graph()
        self.clear()
        # Now to do a 2d input 1d output
        # self.surface_deriv_graph()
        # self.clear()

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

with tempconfig({"quality": "low_quality", "preview": True}):
    # scene = Main()
    # scene.render()
    threeD_scene = Main_3D()
    threeD_scene.render()
