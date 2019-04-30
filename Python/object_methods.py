"""
### Interesting Python things to know ###

Things that I don't normally use, related to OBJECTS

"""

if __name__ == "__main__":



    # =========================================================================== #

    ### Types of Methods (method, classmethod, staticmethod)

    x = 20
    print('\nDifference between METHOD, CLASSMETHOD, STATICMETHOD')
    class Add(object):
        x = 9   # CLASS variable

        def __init__(self, x):
            self.x = x  # INSTANCE variable

        # Normal method operates with the INSTANCE variable
        def AddMethod(self, y):
            print("Method (x + %d):" %y, self.x + y)

        # CLASSmethod operates with the CLASS variable
        @classmethod
        def AddClassmethod(cls, y):
            print("Classmethod: (x + %d):" %y, cls.x + y)

        # STATICmethod operates with variables in the main program (outside the class)
        @staticmethod
        def StaticMethod(y):
            print("Static: (x + %d):" %y, x + y)

    add_object = Add(x=4)
    add_object.AddMethod(10)
    add_object.AddClassmethod(10)
    add_object.StaticMethod(10)

    # Static methods are useful as methods that do not depend on any of the features of the object
    # such as unit conversions and other operations with outside variables / parameters

    # =========================================================================== #
    print('\nDECORATORS')

    ### The DECORATOR
    def addOne(function):
        # Add one receives a function f(x), retrieves the x and operates with it
        def addOneInside(*args, **kwargs):
            return function(*args, **kwargs) + 1
        print('Calling the function addOne')
        return addOneInside

    # Function to be DECORATED
    def function(x):
        print('Calling the function square(x)')
        return x**2

    print(function(2))
    # Now the function is DECORATED with addONE
    decorated = addOne(function)
    print(decorated(2))

    print('\nYou can DECORATE the function by saying:')
    print('function=decorator(function) and use it')
    function = addOne(function)
    print(function(2))

    print('\nYou can further simplify it by using @decorator in the definition')
    @addOne
    def function(x):
        print('Calling the function square(x)')
        return x**2

    print(function(2))