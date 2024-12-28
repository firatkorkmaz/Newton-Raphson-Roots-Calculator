# Find Roots with Newton-Raphson Method

import re
import math
import cmath
import numpy as np
from inspect import signature


# Main program to find roots
class RootsCalculator:
    class MyComplex(complex):
        def __new__(cls, cmp, tol=1e-4, rnd=8):
            obj = super().__new__(cls, cmp)
            obj.tol = tol    # Custom tolerance
            obj.rnd = rnd    # Rounding precision
            return obj

        def __repr__(self):
            real = 0 if abs(self.real) < self.tol else self.real
            imag = 0 if abs(self.imag) < self.tol else self.imag
            real = round(real) if math.isclose(round(real), real, abs_tol=self.tol) else round(real, self.rnd)
            imag = round(imag) if math.isclose(round(imag), imag, abs_tol=self.tol) else round(imag, self.rnd)
            
            if imag == 0:
                return str(real)
            elif real == 0:
                return str(imag) + 'j'
            else:
                return super().__repr__()

    
    @staticmethod
    # Convert python-syntax equations to text-syntax equations
    def python_to_text(input_data):
        def apply(expression):
            expression = re.sub(r'([a-zA-Z])\s*\*\s*([a-zA-Z])', r'\1\2', expression)
            expression = re.sub(r'(\d|\w)\s*\*\s*(\d|\w)', r'\1\2', expression)
            expression = re.sub(r'(\))\s*\*\s*(\d|\w)', r'\1\2', expression)
            expression = re.sub(r'(\d|\w)\s*\*\s*(\()', r'\1\2', expression)
            expression = re.sub(r'(\))\s*\*\s*(\()', r'\1\2', expression)
            expression = re.sub(r'\*\*', '^', expression)
            expression = re.sub(r'\s+', '', expression)
            return expression

        if isinstance(input_data, str):
            return apply(input_data)
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return [apply(item) for item in input_data]
        else:
            raise ValueError("Input must be a string or a list of strings!")
    
    
    @staticmethod
    # Convert text-syntax equations to python-syntax equations
    def text_to_python(input_data):
        def apply(expression):
            # Replace braces with parentheses
            expression = expression.replace('{', '(').replace('}', ')')
            expression = expression.replace('\left', '').replace('\right', '')
            expression = expression.replace('\ln(', '_').replace('ln(', '_')
            expression = expression.replace('\log(', '&').replace('log(', '&')
            expression = expression.replace('\sqrt(', '~').replace('sqrt(', '~')
            expression = expression.replace('\pow(', '¨').replace('pow(', '¨')
            expression = expression.replace('\exp(', 'é').replace('exp(', 'é')
            expression = expression.replace('\sin(', 'á').replace('sin(', 'á')
            expression = expression.replace('\cos(', 'ä').replace('cos(', 'ä')
            expression = expression.replace('\tan(', 'ë').replace('tan(', 'ë')

            expression = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expression)
            expression = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expression)
            expression = re.sub(r'([a-zA-Z])(\()', r'\1*\2', expression)
            expression = re.sub(r'(\))([a-zA-Z\d])', r'\1*\2', expression)
            expression = re.sub(r'(\))(\()', r'\1*\2', expression)
            expression = re.sub(r'(\)|[a-zA-Z])(_|&|~|¨|é|á|ä|ë)', r'\1*\2', expression)
            expression = re.sub(r'\^', '**', expression)
            expression = re.sub(r'\s+', '', expression)

            expression = expression.replace('_', 'ln(')
            expression = expression.replace('&', 'log(')
            expression = expression.replace('~', 'sqrt(')
            expression = expression.replace('¨', 'pow(')
            expression = expression.replace('é', 'exp(')
            expression = expression.replace('á', 'sin(')
            expression = expression.replace('ä', 'cos(')
            expression = expression.replace('ë', 'tan(')
            return expression

        # Handle string or list inputs
        if isinstance(input_data, str):
            return apply(input_data)
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return [apply(item) for item in input_data]
        else:
            raise ValueError("Input must be a string or a list of strings!")

            
    @staticmethod
    # Convert python-syntax equations to lambda functions
    def python_to_lambda(input_data):
        def apply(equation):
            # Remove spaces from equation
            equation = equation.replace(" ", "")
            
            # Handle equality (two sides of the equation)
            equals_index = equation.find('=')
            if equals_index != -1:
                left_part = equation[:equals_index].strip()
                right_part = equation[equals_index + 1:].strip()

                # Formulate the equation in the form (left_part) - (right_part)
                equation = f"({left_part})-({right_part})"
            
            # List of known mathematical functions to exclude
            math_func = {"ln", "log", "sin", "cos", "tan", "exp", "sqrt", "pow"}

            # Find potential variable names using regex
            identifiers = set(re.findall(r"[a-zA-Z_]\w*", equation))

            # Exclude known mathematical functions
            variables = list(identifiers - math_func)

            # Check for any extra variables other than 'x', 'y', or 'e'
            extra_vars = set(variables) - {"x", "y", "e"}

            if extra_vars:
                return f"Error: Only 'x' and 'y' are allowed. Found extra variable(s): {', '.join(extra_vars)}"
            else:
                return f"lambda x, y: {equation}"

        # Handle both single string or list of strings input
        if isinstance(input_data, str):
            return apply(input_data)
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return [apply(item) for item in input_data]
        else:
            raise ValueError("Input must be a string or a list of strings!")

        
    # Initialize RootsCalculator program
    def __init__(self, func=None):
        if func:
            function = self.python_to_lambda(self.text_to_python(func))
        else:
            function = self.python_to_lambda(self.text_to_python("y=x"))
        self.func_f = eval(function)
        self.is_implicit = len(signature(self.func_f).parameters) > 1
        self.h = 1e-6
        self.max_iter = 1000
        self.tolerance = 1e-12
        self.tol = 1e-4
        self.rnd = 8
        
    
    # Partial derivative function for implicit functions
    def partial_derivative(self, x, y=None, respect_to="x"):
        # Compute the partial derivative of the function with respect to x or y
        if not self.is_implicit:
            raise ValueError("Partial derivatives are not required for explicit functions.")
        
        if respect_to == "x":
            return (self.func_f(x + self.h, y) - self.func_f(x - self.h, y)) / (2 * self.h)
        elif respect_to == "y":
            return (self.func_f(x, y + self.h) - self.func_f(x, y - self.h)) / (2 * self.h)

    
    # Derivative function for explicit equations
    def derivative(self, x):
        # Compute the derivative for explicit functions
        if self.is_implicit:
            raise ValueError("Explicit derivatives are not required for implicit functions.")
        return (self.func_f(x + self.h) - self.func_f(x - self.h)) / (2 * self.h)

    
    def range_values(self, start, end):
        tune = 0.1
        range_list = range(10 * start, 10 * end + 1)
        real_ = [tune * x for x in range_list]
        pos_i = [tune * x * (1 + 1j) for x in range_list]
        neg_i = [tune * x * (1 - 1j) for x in range_list]
        initial_values = real_ + pos_i + neg_i
        return initial_values
    
    
    # Root finder function
    def roots_f(self, x_=None, y_=None, r_=(-10, 10)):
        r1, r2 = r_
        roots = []; roots_ = [];
        roots_y = []; roots_x = [];
        initial_values = self.range_values(r1, r2 + 1)
        
        if x_ is not None:
            for x in x_:
                for y in initial_values:
                    for _ in range(self.max_iter):
                        try:
                            f_xy = self.func_f(x, y)
                            dfdy = self.partial_derivative(x, y, respect_to="y")

                            # Avoid division by zero
                            if dfdy == 0:
                                break

                            # Convergence algorithm
                            y_new = y - f_xy / dfdy

                            # Convergence check
                            if abs(y_new - y) < self.tolerance:
                                break

                            y = y_new
                        except ValueError:
                            continue
                            
                        except ZeroDivisionError:
                            break

                    # Round the results to avoid duplicate roots
                    real = round(y.real) if math.isclose(round(y.real), y.real, abs_tol=self.tol) else round(y.real, self.rnd)
                    imag = round(y.imag) if math.isclose(round(y.imag), y.imag, abs_tol=self.tol) else round(y.imag, self.rnd)
                    y_new = complex(real, imag)
                    
                    try:
                        # Now check if the function evaluated at (x, y_new) is close to zero before appending
                        if cmath.isclose(self.func_f(x, y_new), 0, abs_tol=self.tol) and not any(cmath.isclose(x, r[0], abs_tol=self.tolerance) and cmath.isclose(y_new, r[1], abs_tol=self.tolerance) for r in roots):
                            roots_y.append((x, y_new))
                    except ValueError:
                        continue
            
            roots_y = [(RootsCalculator.MyComplex(root[0], self.tol, self.rnd), RootsCalculator.MyComplex(root[1], self.tol, self.rnd)) for root in roots_y]
            roots_y = sorted(roots_y, key=lambda root: (root[1].imag != 0, (root[0].real, (root[1].real, root[1].imag))))
        
            
        if y_ is not None:
            for y in y_:
                for x in initial_values:
                    for _ in range(self.max_iter):
                        try:
                            f_xy = self.func_f(x, y)
                            dfdx = self.partial_derivative(x, y, respect_to="x")

                            # Avoid division by zero
                            if dfdx == 0:
                                break

                            # Convergence algorithm
                            x_new = x - f_xy / dfdx

                            # Convergence check
                            if abs(x_new - x) < self.tolerance:
                                break

                            x = x_new
                        except ValueError:
                            continue
                        
                        except ZeroDivisionError:
                            break

                    # Round the results to avoid duplicate roots
                    real = round(x.real) if math.isclose(round(x.real), x.real, abs_tol=self.tol) else round(x.real, self.rnd)
                    imag = round(x.imag) if math.isclose(round(x.imag), x.imag, abs_tol=self.tol) else round(x.imag, self.rnd)
                    x_new = complex(real, imag)
                    
                    try:
                        # Now check if the function evaluated at (x_new, y) is close to zero before appending
                        if cmath.isclose(self.func_f(x_new, y), 0, abs_tol=self.tol) and not any(cmath.isclose(x_new, r[0], abs_tol=self.tolerance) and cmath.isclose(y, r[1], abs_tol=self.tolerance) for r in roots):
                            roots_x.append((x_new, y))
                    except ValueError:
                        continue
            
            roots_x = [(RootsCalculator.MyComplex(root[0], self.tol, self.rnd), RootsCalculator.MyComplex(root[1], self.tol, self.rnd)) for root in roots_x]
            roots_x = sorted(roots_x, key=lambda root: (root[0].imag != 0, (root[1].real, (root[0].real, root[0].imag))))
        
        
        # Listing roots by parameter-based order 
        roots = roots_y + roots_x
        for root in roots:
            x, y = root
            if not any(cmath.isclose(x, r[0], abs_tol=self.tol) and cmath.isclose(y, r[1], abs_tol=self.tol) for r in roots_):
                roots_.append((x, y))
        
        return roots_
    
    
    def function(self, x, y):
        result = self.func_f(x, y)
        return self.MyComplex(result, self.tol, self.rnd)
        
    
    @staticmethod
    # Function to run main program
    def find_roots(equation_list=None, **kwargs):
        
        if equation_list is None:
            equation_list = ["y=x"]
            
        elif isinstance(equation_list, str):
            equation_list = [equation_list]
        
        
        # Extract x_ and y_ from kwargs or set to default if not provided
        x = kwargs.get('x', None)
        y = kwargs.get('y', None)
        
        # Convert input x_ and y_ to lists if necessary
        if x is not None and not isinstance(x, list):
            x = [x]
        if y is not None and not isinstance(y, list):
            y = [y]

        
        for i, equation in enumerate(equation_list, start=1):
            
            if "=" not in equation:
                #equation += "=0"
                if "x" in equation and "y" not in equation:
                    equation = "y=" + equation
                elif "y" in equation and "x" not in equation:
                    equation = equation + "=x"
                else:
                    equation = equation + "=0"
                
                
            # Instantiating RootsCalculator
            calculator = RootsCalculator(equation)
            
            
            x_ = None; y_ = None
            
            if "x" in equation and "y" not in equation:
                y_ = [0]
            elif "y" in equation and "x" not in equation:
                x_ = [0]
            else:
                if x is None and y is None:
                    y_ = [0]
                else:
                    x_ = x; y_ = y
            
            # Get the roots with input variable lists
            roots = calculator.roots_f(x_, y_)

            
            # Print the results
            print(f"Equation-{i:02}: {calculator.python_to_text(calculator.text_to_python(equation))}")
            print(f"F = {calculator.python_to_lambda(calculator.text_to_python(equation))}")
            
            
            if "x" in equation and "y" not in equation:
                print(f"Find x values for F(x, y) = 0, where y = 0")
            elif "y" in equation and "x" not in equation:
                print(f"Find y values for F(x, y) = 0, where x = 0")
            else:
                if x is None and y is None:
                    print(f"Find x values for F(x, y) = 0, where y = 0")
                else:
                    if x_ is not None:
                        print(f"Find y values for F(x, y) = 0, where x = {x_}")
                    if y_ is not None:
                        print(f"Find x values for F(x, y) = 0, where y = {y_}")  
            print()
            
            
            for j, root in enumerate(roots, start=1):
                _x, _y = root
                print(f"{j}. F({_x}, {_y}) = {calculator.function(_x, _y)}")
            
            if i < len(equation_list):
                print("\n###################################################\n")
                
################################################################################

# Main execution pipeline
if __name__ == "__main__":
    import re
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse input equations and initial variables.")
    parser.add_argument("--equation", type=str, nargs='+', required=False, default=None, help="Enter each equation in double quotation marks, separated by a space (e.g., \"f1\" \"f2\")")
    parser.add_argument("--x", type=int, nargs='+', required=False, default=None, help="Enter x values to find y values, separated by a space (e.g., -1 0 1)")
    parser.add_argument("--y", type=int, nargs='+', required=False, default=None, help="Enter y values to find x values, separated by a space (e.g., -2 1 0)")
    args = parser.parse_args()
    
    e = cmath.e; pi = cmath.pi
    sin = cmath.sin; cos = cmath.cos
    tan = cmath.tan; exp = cmath.exp
    ln = cmath.log; sqrt = cmath.sqrt
    Complex = RootsCalculator.MyComplex
    
    def log(x, base=None):
        if base is None:
            return cmath.log(x)
        else:
            if isinstance(x, complex) or isinstance(base, complex):
                return cmath.log(x) / cmath.log(base)
            else:
                return math.log(x, base)


    # Check for missing arguments and prompt the user
    if args.equation is None and args.x is None and args.y is None:
        print()
        user_input = input("Enter equations (e.g., \"y=2x\", \"x^3=8\"): ").strip()
        user_input = re.split(r',\s*|\s+', user_input.replace('"', '').replace("'", ""))
        args.equation = [item for item in user_input if item]
        
        print()
        user_input = input("Enter x values to find y values (e.g., -1, 2): ").strip()
        user_input = re.split(r',\s*|\s+', user_input)
        args.x = [int(item) for item in user_input if item]
        if len(args.x) == 0:
            args.x = None
        
        print()
        user_input = input("Enter y values to find x values (e.g., -2, 0): ").strip()
        user_input = re.split(r',\s*|\s+', user_input)
        args.y = [int(item) for item in user_input if item]
        if len(args.y) == 0:
            args.y = None
            
            
    print()
    RootsCalculator.find_roots(args.equation, x=args.x, y=args.y)
