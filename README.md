# Newton-Raphson Roots Calculator
A program written in Python to find the roots of input equations with two variables; x and y.

## General Information
This program is intended to find the roots of any input equations with either one (x) or two (x, y) variables.

Examples:
```
x^2-4x=0
y=2x^4-3x^2
2x^3+x^2-y^2=0
```
Both python-syntax and text-syntax equation inputs are supported.

Text-syntax equation example:
```
y=2x^4-3x^2+x
```
Python-syntax equation example:
```
y=2*x**4-3*x**2+x
```

There are 2 program files in this project, the first one is a Jupyter Notebook file which has the main program with some details explained in additional notebook cells, and the second file is pure Python program that can be run easily through terminal:

1. **RootsCalculator.ipynb**: Notebook version of the program.
2. **RootsCalculator.py**: Python file version of the program.

## Setup & Run
The python file can easily be run with such commands (NOTE: When the equations are given as input arguments, each equation must be in its own double quotation marks):
```
python RootsCalculator.py
```
> The program will request input equations to analyze, input x values for which to find y values, and input y values for which to find x values.

```
python RootsCalculator.py --equation "y=2x^4-3x^2+x"
```
> The program will not ask for any x or y values, it will find: x values of the equation where y = 0

 ```
python RootsCalculator.py --equation "x^2-4y=0" --x -1
```
> The program will not ask for any y values, it will find: y values of the equation where x = -1

 ```
python RootsCalculator.py --equation "y=2x^4-3x^2+x" "x^2-4y=0" --y 0 1
```
> The program will not ask for any x values, it will find: x values for y = 0, and y = 1, per equation

 ```
python RootsCalculator.py --equation "y=2x^4-3x^2+x" "x^2-4y=0" --x -1 --y 1
```
> The program will not ask for anything else, it will find: y values for x = -1, and x values for y = 1, per equation

```
# Running the program from Jupyter Notebook with a group of different equation samples

equation_list = [
    "y=x^2",       # Style: f(y)=f(x)
    "x^2-2x=0",    # Style: f(x)=0
    "x^2-2x",      # Style: f(x) >> "y=" will be added
    "y^2+4y=0",    # Style: f(y)=0
    "y^2+4y",      # Style: f(y) >> "=x" will be added
    "x^2-4y=0",    # Style: f(x,y)=0
    "x^2-4y",      # Style: f(x,y)  "=0" will be added
]

# Run the program with a list of equations
RootsCalculator.find_roots(equation_list, x=[-1, 0], y=1)
```

> This equation list to run the program with will be converted to the equations below just before the root calculation process:
```
Equation-01: y=x^2
Equation-02: x^2-2x=0
Equation-03: y=x^2-2x
Equation-04: y^2+4y=0
Equation-05: y^2+4y=x
Equation-06: x^2-4y=0
Equation-07: x^2-4y=0
```

### Example Program Run

<img title="Running the Roots Calculator" src="https://github.com/firatkorkmaz/Newton-Raphson-Roots-Calculator/blob/main/images/RootsCalculator.png">
 
