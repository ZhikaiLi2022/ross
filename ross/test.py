#---------------------------------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------------------------------------------------------------
# class
class Person:
    # attribute fields
    name = 'William'
    age = 45
    #__age =  #44 The arguments will unable to mmobilize if you add '__' in front of arguments
    # method
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print("Hi, my name is " + self.name+"    |")
        print('''--------------------------''')
        
# Create an Object
p1 = Person()
# Call the method
p1.name = "Enoch"
p1.greet() # instance

# Modify Object Properties
p1.age = 40 #you can to modify and call parameters
#p1.__age = 40 # you are unable to modify and call parameters
print(p1.age)

# Delete Object Properties
del p1.age

#Delete Objects
#del p1
print(p1.age)

#---------------------------------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------------------------------------------------------------
class Person:
    def __init__(self): #iniitialization
        self.name = 'Alice'
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print("Hi, my name is " + self.name+"    |")
        print('''--------------------------''')
p1 = Person()
p1.greet()

class Person:
    def __init__(self, init_name):
        self.name = init_name
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print("Hi, my name is " + self.name+"    |")
        print('''--------------------------''')
p1 = Person("David")
p1.greet()

#---------------------------------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------------------------------------------------------------
# inheritance, Polymorphism
class Animal():
    def __init__(self, name):
        self.name = name
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print('Hello, I am an %s. ' % self.name +" |")
        print('''--------------------------''')
class Dog():
    def __init__(self, name):
        self.name = name
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print('WangWang.., I am a %s. ' % self.name +"|")
        print('''--------------------------''')

class Dog(Animal):
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print('WangWang.., I am a %s. ' % self.name +"|")
        print('''--------------------------''')
animal = Animal('animal')
animal.greet()
dog = Dog('dog')
dog.greet()

#---------------------------------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------------------------------------------------------------
class Dog(Animal):
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print('WangWang.., I am a %s. ' % self.name +"|")
        print('''--------------------------''')
    def run(self):
        print('''--------------------------''')
        print('I am running!           |')
        print('''--------------------------''')

dog = Dog('dog')
dog.greet()
dog.run()

#---------------------------------------------------------------------------------------------------
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#---------------------------------------------------------------------------------------------------
#Polymorphism
class Animal():
    def __init__(self, name):
        self.name = name
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print(f'Hello, I am an {self.name} .'+" |")
        print('''--------------------------''')

class Dog(Animal):
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print(f'WangWang.., I am a {self.name} .'+"|")
        print('''--------------------------''')

class Cat(Animal):
    def greet(self): # the self is instance itself.
        print('''--------------------------''')
        print(f'MiaoMiao.., I am a {self.name} .'+"|")
        print('''--------------------------''')
def hello(animal):
    animal.greet()
'''
def hello(dog):
    pass
def hello(cat):
    pass
'''
dog = Dog('dog')
hello(dog)
cat = Cat('cat')
hello(cat)
