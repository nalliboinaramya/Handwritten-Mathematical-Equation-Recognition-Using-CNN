import sympy as sym
from sympy import *
def solve_equation(equation):
    try:
        # print(equation)
        # print("1")
        if 'x' in equation and '=' in equation:
            x = sym.symbols('x')
            left, right = equation.split("=")
            # print("6")
            eq = left+'-'+right
            # print('---------------')
            result = sym.solve(eq, (x))

            return result
        elif 'x' in equation and '=' not in equation:
            x = sym.symbols('x')
            result = sym.solveset(equation, x)
            return result
        
        elif 'y' in equation and '=' in equation:
            y = sym.symbols('y')

            left, right = equation.split("=")
            # print("6")
            eq = left+'-'+right
            # print('---------------')
            result = sym.solve(eq, (y))

            return result
        elif 'y' in equation and '=' not in equation:
            y=sym.symbols('y')
            result = sym.solveset(equation,y)
            return result
        
        elif 'x' in equation and 'y' in equation and '=' in equation :
            x,y = sym.symbols('x,y')

            left, right = equation.split("=")
            # print("6")
            eq = left+'-'+right
            # print('---------------')
            result = sym.solve(eq, (x,y))

            return result
        
        elif '=' in equation and 'sin' in equation or 'tan' in equation or 'cos' in equation:
            if 'x' in equation and 'y' in equation:
                x,y = symbols('x,y')
                eq_sympy = sympify(equation)
                result = solve((eq_sympy), (x,y))
                return result
            elif 'x' in equation:
                x = symbols('x')
                eq_sympy = sympify(equation)
                result = solve((eq_sympy), x)
                return result
            elif 'y' in equation:
                y = symbols('y')
                eq_sympy = sympify(equation)
                result = solve((eq_sympy), y)
                return result
            else:
                pass
        else:
            pass
        # print("2")

        if "=" not in equation:

            if 'sin' in equation or 'tan' in equation:
                eq = sympify(equation)
                result = eq.evalf()
                return result
            
            # print("3")
            result = sym.simplify(equation)
            # print("4")
            # print(f"Result: {result}")
            return result

        else:
            # print("5")
            left, right = equation.split("=")
            # print("6")
            # eq = left+'-'+right
            # print('---------------')
            result = sym.solve(left,right)
            # result = sym.solve(left + "-" + right, x)

            if len(result) == 0:
                # print("No solution found.")
                return "No solution found." 
            else:
                # print("Solution(s):")
                return result

    except (ValueError, TypeError, AttributeError, RuntimeError, SyntaxError) as e:
        # print(f"Error: {e}")
        return str('Wrong equation prediction')